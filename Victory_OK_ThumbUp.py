import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 定义手指的索引
FINGER_TIPS = [4, 8, 12, 16, 20]

# 定义手势枚举
class Gesture:
    V = "V Gesture"
    OK = "OK Gesture"
    THUMBS_UP = "Thumbs Up"
    UNKNOWN = "Unknown"

# 手势识别函数
def detect_gesture(hand_landmarks):
    # 计算手指是否伸直
    fingers = []
    # 拇指
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # 其他手指
    for tip in FINGER_TIPS[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    # 判断手势
    # 点赞：只有拇指伸直
    if fingers == [1, 0, 0, 0, 0]:
        return Gesture.THUMBS_UP
    # V 手势：食指和中指伸直，其余手指弯曲
    elif fingers == [0, 1, 1, 0, 0]:
        return Gesture.V
    # OK 手势：拇指和食指形成圈，其余手指伸直
    else:
        # 计算拇指和食指的距离
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        if distance < 0.05 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            return Gesture.OK
    return Gesture.UNKNOWN

# 烟花特效类
class Firework:
    def __init__(self, position):
        self.position = position
        self.particles = []
        self.create_particles()
    
    def create_particles(self):
        for _ in range(20):  # 每个烟花20个粒子
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            self.particles.append({
                'position': list(self.position),
                'velocity': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': 4,
                'color': (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
                'life': 50
            })
    
    def update(self, frame):
        for particle in self.particles[:]:
            # 更新位置
            particle['position'][0] += particle['velocity'][0]
            particle['position'][1] += particle['velocity'][1]
            # 重力效果
            particle['velocity'][1] += 0.1
            # 减小生命值
            particle['life'] -= 1
            # 绘制粒子
            cv2.circle(frame, (int(particle['position'][0]), int(particle['position'][1])), particle['size'], particle['color'], -1)
            # 移除生命耗尽的粒子
            if particle['life'] <= 0:
                self.particles.remove(particle)

# 霓虹特效类
class NeonEffect:
    def __init__(self, position):
        self.position = position
        self.max_radius = 150
        self.min_radius = 100
        self.radius = self.min_radius
        self.growing = True
        self.color = (0, 255, 255)  # 青色霓虹
        self.thickness = 4
        self.duration = 3.0  # 持续3秒
        self.start_time = time.time()
    
    def update(self, frame):
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            return False  # 特效结束
        
        # 脉动半径
        if self.growing:
            self.radius += 2
            if self.radius >= self.max_radius:
                self.growing = False
        else:
            self.radius -= 2
            if self.radius <= self.min_radius:
                self.growing = True
        
        # 绘制霓虹圆环
        cv2.circle(frame, self.position, self.radius, self.color, self.thickness)
        # 添加发光效果
        for i in range(1, 4):
            overlay_color = (self.color[0], self.color[1], self.color[2], max(255 - i * 60, 0))
            cv2.circle(frame, self.position, self.radius + i * 10, self.color, 2)
        
        return True

# 鼓掌特效类
class ClapEffect:
    def __init__(self, position):
        self.position = position
        self.start_time = time.time()
        self.duration = 1.0  # 持续1秒
    
    def update(self, frame):
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            return False  # 特效结束
        alpha = 1.0 - (elapsed / self.duration)
        size = int(100 * alpha)
        if size > 0:
            cv2.putText(frame, "WoW", self.position, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
        return True

# 特效管理类
class EffectsManager:
    def __init__(self):
        self.fireworks = []
        self.neon_effects = []
        self.clap_effects = []
    
    def trigger_firework(self, position):
        self.fireworks.append(Firework(position))
    
    def trigger_neon(self, position):
        self.neon_effects.append(NeonEffect(position))
    
    def trigger_clap(self, position):
        self.clap_effects.append(ClapEffect(position))
    
    def update(self, frame):
        # 更新烟花
        for firework in self.fireworks[:]:
            firework.update(frame)
            if not firework.particles:
                self.fireworks.remove(firework)
        
        # 更新霓虹特效
        for neon in self.neon_effects[:]:
            active = neon.update(frame)
            if not active:
                self.neon_effects.remove(neon)
        
        # 更新鼓掌
        for clap in self.clap_effects[:]:
            active = clap.update(frame)
            if not active:
                self.clap_effects.remove(clap)

# 特效显示函数
def show_effect(frame, gesture, effects_manager, hand_landmarks):
    # 根据手势类型触发对应特效
    if gesture == Gesture.V:
        # 获取手腕坐标作为烟花位置
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        h, w, _ = frame.shape
        position = (int(wrist.x * w), int(wrist.y * h))
        effects_manager.trigger_firework(position)
    elif gesture == Gesture.OK:
        # 获取手腕坐标作为霓虹特效中心
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        h, w, _ = frame.shape
        position = (int(wrist.x * w), int(wrist.y * h))
        effects_manager.trigger_neon(position)
    elif gesture == Gesture.THUMBS_UP:
        # 获取手腕坐标作为鼓掌位置
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        h, w, _ = frame.shape
        position = (int(wrist.x * w), int(wrist.y * h))
        effects_manager.trigger_clap(position)

# 主函数
def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    gesture = Gesture.UNKNOWN
    effects_manager = EffectsManager()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)  # 镜像翻转
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                current_gesture = detect_gesture(hand_landmarks)
                
                if current_gesture != Gesture.UNKNOWN:
                    # 如果检测到新的手势，触发对应特效
                    if current_gesture != gesture:
                        gesture = current_gesture
                        show_effect(frame, gesture, effects_manager, hand_landmarks)
                    
                    # 绘制关键点为红色，连接线为绿色且加粗
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_draw.DrawingSpec(color=(0, 255, 0), thickness=4),
                                          mp_draw.DrawingSpec(color=(0, 0, 255), thickness=6))
                else:
                    gesture = Gesture.UNKNOWN
                    # 默认绘制方式
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
                                          mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2))
        
        # 更新并显示特效
        effects_manager.update(frame)
        
        # 显示帧率
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("Hand Gesture Recognition", frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
