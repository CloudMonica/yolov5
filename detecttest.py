import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox

# 参数设置
weights_path = 'best.pt'  # 模型权重文件路径
# device = 'cpu'  # 设备选择（例如 '0'、'0,1,2,3' 或 'cpu'）
img_size = 640 # 图像大小

# 加载模型
# device = select_device(device)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = attempt_load(weights_path)
model.to(device)
img_size = check_img_size(img_size, s=model.stride.max())  # 检查图像大小

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 从摄像头获取图像
    ret, frame = cap.read()

    # 图像预处理
    img = letterbox(frame, new_shape=img_size)[0]  # 调整图像大小并进行填充
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # 推理
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推理
    with torch.no_grad():
        pred = model(img)[0]

    # NMS后处理
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 显示图像
    cv2.imshow('YOLOv5 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()