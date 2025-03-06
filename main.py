from ultralytics import YOLO
import cv2
import time
import torch
import torch.nn as nn
import numpy as np


class LSTMModel_expanded(nn.Module):
    def __init__(self):
        super(LSTMModel_expanded, self).__init__()

        # Lớp LSTM với hidden_size lớn hơn và nhiều layers hơn
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=256,
            num_layers=3,  # Giảm số lượng layers để tăng hiệu quả
            batch_first=True,
            dropout=0.3,  # Thêm dropout trong LSTM
        )

        # Fully connected layers với số lượng units lớn hơn
        self.fc1 = nn.Linear(256, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)  # Dự đoán 3 class

        # Thay BatchNorm bằng LayerNorm
        self.layernorm1 = nn.LayerNorm(256)
        self.layernorm2 = nn.LayerNorm(128)
        self.layernorm3 = nn.LayerNorm(64)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Forward pass qua LSTM
        out, (h_n, c_n) = self.lstm(x)  # out có kích thước [batch_size, seq_len, hidden_size]

        # Chỉ lấy đầu ra của bước cuối cùng trong chuỗi
        out = out[:, -1, :]

        # Pass qua fully connected layers
        out = self.fc1(out)
        out = self.layernorm1(out)
        out = torch.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.layernorm2(out)
        out = torch.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.layernorm3(out)
        out = torch.relu(out)
        out = self.dropout(out)

        out = self.fc4(out)

        return out


# Load model
model = LSTMModel_expanded()
model.load_state_dict(torch.load("lstmmodel.pth", map_location=torch.device('cpu'))) # Load model từ file, chuyển model về CPU
model.eval()  # Chuyển model sang chế độ inference
# Tải mô hình YOLO
model_yolo = YOLO("best.pt")# Load model YOLOv8 pose

# Mở camera (camera mặc định, ID=0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

frame_counter = 0  # Counter for frame print
frame_list = []
a = 0 
while True:
    # Đặt thời gian bắt đầu cho mỗi lần dự đoán
    start_time = time.time()

    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera")
        break

    # Lấy kích thước ban đầu của khung hình
    original_height, original_width = frame.shape[:2]

    # Resize khung hình về kích thước 640x640 cho mô hình YOLO
    resized_frame = cv2.resize(frame, (640, 640))

    # Dự đoán trên khung hình đã resize và lưu kết quả vào đối tượng `results`
    results = model_yolo(resized_frame, imgsz=640)
    arrs = results[0].keypoints.data[0]

    # Tạo danh sách lưu tọa độ keypoints của khung hình hiện tại
    frame_keypoints = []
    for arr in arrs:
        point = arr[:2].cpu().numpy()  # Chuyển tensor thành numpy array
        # Điều chỉnh lại tọa độ keypoints theo kích thước khung hình gốc
        x = int(point[0] * original_width / 640)
        y = int(point[1] * original_height / 640)
        frame_keypoints.extend([x, y])  # Thêm cặp (x, y) vào danh sách
        # Vẽ điểm keypoint lên khung hình gốc
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    if len(frame_keypoints) == 0: continue
    
    frame_list.append(frame_keypoints)
    frame_counter += 1
    
# After accumulating 10 frames in `frame_list`
    if frame_counter == 10:
        # Convert list to numpy array and reshape to 1x10x8
        array_10x8_np = np.array(frame_list)  # Convert to array before reshaping
        frame_list_arr = array_10x8_np.reshape(1, 10, 8)
        frame_list_tensor = torch.from_numpy(frame_list_arr).float()
        # Print to verify the shape
        print("Shape after reshape:", frame_list_tensor.shape)  # Should print (1, 10, 8)

        # Run the model prediction with `frame_list_arr` here
        # results = model(torch.tensor(frame_list_arr).float())
        
        # Reset the counter and clear the list after processing
        frame_list = []
        frame_counter = 0

        results = model(frame_list_tensor)

        # Get the index of the class with the highest score
        predicted_class = torch.argmax(results, dim=1)
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(results)
        max_index = torch.argmax(results, dim=1)
        a =   max_index[0]
        print(max_index)
    if (a == 0):
        cv2.putText(frame, "Binh Thuong", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,cv2.LINE_AA)
    elif (a== 1):
        cv2.putText(frame, "Lech", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,cv2.LINE_AA)
    elif (a== 2):
        cv2.putText(frame, "Bi keo", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,cv2.LINE_AA)
    
    # Hiển thị khung hình với các điểm keypoints đã được vẽ
    cv2.imshow("Camera Feed with Keypoints", frame)

    # Đợi khoảng thời gian để đảm bảo 200ms mỗi lần dự đoán
    time.sleep(max(0, 0.2 - (time.time() - start_time)))

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()

print("Keypoints đã được lưu vào file keypoints.txt.")