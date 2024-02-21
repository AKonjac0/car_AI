import cv2
from ultralytics import YOLO


def yolo(video_path):
    model = YOLO('yolov8l.pt')
    cap = cv2.VideoCapture(video_path)

# 视频帧循环
    cnt = [0, 0, 0, 0, 0, 0, 0, 0]
    while cap.isOpened():
    # 读取一帧图像
        success, frame = cap.read()
        for i in range(0, 24):
            success, frame = cap.read()
        if success == 0:
            break
        if success:
            # 在帧上运行YOLOv8跟踪，persist为True表示保留跟踪信息，conf为0.3表示只检测置信值大于0.3的目标
            results = model.track(frame, conf=0.3, persist=True)
            # 遍历该帧的所有目标
            for box in results[0].boxes.data:
                if box[-1] == 2:  # 目标为小汽车
                    cnt[2] += 1
                elif box[-1] == 5:
                    cnt[5] += 1
                elif box[-1] == 7:
                    cnt[7] += 1
                else:
                    cnt[0] += 1

    tot = 0
    pos = -1
    m = 0
    for i in cnt:
        if i > m:
            m = i
            pos = tot
        tot += 1

    if pos == 2:
        return "passenger car"
    elif pos == 7:
        return "truck"
    elif pos == 5:
        return "bus"
    elif pos == 0:
        return "nothing"
    # 释放视频捕捉对象，并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()

