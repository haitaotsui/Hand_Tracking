import cv2
import mediapipe as mp
import numpy as np
import random
import time

# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化 OpenCV 摄像头
cap = cv2.VideoCapture(0)

# 检测“V”手势
def is_victory_gesture(hand_landmarks, frame_width, frame_height):
    keypoints = [(int(p.x * frame_width), int(p.y * frame_height)) for p in hand_landmarks.landmark]
    
    # 食指和中指的指尖
    index_tip = keypoints[8]
    middle_tip = keypoints[12]
    
    # 食指和中指的指根
    index_mcp = keypoints[5]
    middle_mcp = keypoints[9]

    # 检测食指和中指伸直
    is_index_straight = index_tip[1] < index_mcp[1]
    is_middle_straight = middle_tip[1] < middle_mcp[1]

    # 检测拇指、无名指和小指收拢
    thumb_tip = keypoints[4]
    ring_tip = keypoints[16]
    pinky_tip = keypoints[20]

    is_thumb_closed = thumb_tip[1] > index_mcp[1]
    is_ring_closed = ring_tip[1] > middle_mcp[1]
    is_pinky_closed = pinky_tip[1] > middle_mcp[1]

    # 条件：食指和中指伸直，其他手指收拢
    return is_index_straight and is_middle_straight and is_thumb_closed and is_ring_closed and is_pinky_closed

# 绘制烟花特效
def draw_fireworks(frame):
    h, w, _ = frame.shape
    for _ in range(50):  # 生成 50 个随机点
        x = random.randint(0, w)
        y = random.randint(0, h)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(frame, (x, y), random.randint(5, 10), color, -1)
    return frame

# 主循环
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 翻转视频，增强用户体验
    frame = cv2.flip(frame, 1)

    # 转换为 RGB 格式（MediaPipe 要求的输入格式）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手势检测
    results = hands.process(rgb_frame)

    # 检测到手部时处理
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制手部关键点
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 判断是否是“V”手势
            if is_victory_gesture(hand_landmarks, frame.shape[1], frame.shape[0]):
                cv2.putText(frame, "Victory Gesture Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 播放烟花特效
                frame = draw_fireworks(frame)

    # 显示视频
    cv2.imshow("Hand Gesture Recognition with Fireworks", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
