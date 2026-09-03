import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from mp_utils import Timestamper, ensure_model

# === [1. 설정 값] ===
# 원하는 해상도를 요청하되, 실제로는 카메라가 지원하는 대로 작동하게 됩니다.
REQUEST_W = 1280
REQUEST_H = 720

# 고양이와 물고기 크기
cat_w, cat_h = 200, 200
food_w, food_h = 80, 80
l_margin = 50 # 벽에서 떨어질 거리
r_margin = 20

feed_distance = 100 # 먹이 인식 거리

# 이미지 파일 경로
path_cat_hungry = "./images/cat_hungry.png"
path_cat_happy = "./images/cat_happy.png"
path_fish = "./images/fish.png"

# === [2. 초기화] ===
# MediaPipe 1.x 부터는 mp.solutions.hands 가 없어지고 Tasks API 를 사용합니다.
hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
))
timestamper = Timestamper()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_H)

# === [3. 유틸리티 함수] ===
def load_img(path, size, color=(0,0,255)):
    try:
        if not os.path.exists(path): raise FileNotFoundError
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: raise FileNotFoundError
        img = cv2.resize(img, size)
        return img, True
    except:
        img = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        img[:] = color + (255,)
        cv2.putText(img, path.split('.')[0], (10, size[1]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return img, False

# 이미지 로드
img_cat_hungry, _ = load_img(path_cat_hungry, (cat_w, cat_h), (255, 0, 0))
img_cat_happy, _ = load_img(path_cat_happy, (cat_w, cat_h), (0, 255, 255))
img_fish, _ = load_img(path_fish, (food_w, food_h), (0, 0, 255))

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    ov_h, ov_w, ov_c = overlay.shape

    if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
        return background

    bg_x1 = max(x, 0)
    bg_y1 = max(y, 0)
    bg_x2 = min(x + ov_w, bg_w)
    bg_y2 = min(y + ov_h, bg_h)

    ov_x1 = max(0, -x)
    ov_y1 = max(0, -y)
    ov_x2 = ov_x1 + (bg_x2 - bg_x1)
    ov_y2 = ov_y1 + (bg_y2 - bg_y1)

    overlay_cropped = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    
    if ov_c == 4:
        alpha_mask = overlay_cropped[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask
        roi = background[bg_y1:bg_y2, bg_x1:bg_x2]
        for c in range(3):
            roi[:, :, c] = (alpha_mask * overlay_cropped[:, :, c] +
                            alpha_inv * roi[:, :, c])
        background[bg_y1:bg_y2, bg_x1:bg_x2] = roi
    else:
        background[bg_y1:bg_y2, bg_x1:bg_x2] = overlay_cropped[:, :, :3]
    return background

def is_fist(lm_list):
    fingers_folded = 0
    if lm_list[8][2] > lm_list[6][2]: fingers_folded += 1
    if lm_list[12][2] > lm_list[10][2]: fingers_folded += 1
    if lm_list[16][2] > lm_list[14][2]: fingers_folded += 1
    if lm_list[20][2] > lm_list[18][2]: fingers_folded += 1
    return fingers_folded >= 3

# === 변수 초기화 ===
foods = []
score = 0
cat_state = "hungry"
last_fed_time = 0
dragged_idx = -1
initialized = False # 화면 크기에 맞게 위치를 잡았는지 체크하는 변수

# 고양이 위치 (나중에 설정)
cat_x, cat_y = 0, 0 

print("게임 시작! (종료: ESC)")

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1) # 거울 모드
    h, w, c = img.shape    # [중요] 실제 카메라의 크기를 여기서 가져옴

    # === [최초 1회 실행] 실제 화면 크기에 맞춰서 위치 배치 ===
    if not initialized:
        # 1. 고양이 위치: 오른쪽 끝에서 r_margin만큼 띄움, 수직 중앙
        cat_x = w - cat_w - r_margin
        cat_y = (h // 2) - (cat_h // 2)
        
        # 2. 물고기 배치: 왼쪽 벽에서 l_margin만큼 띄움
        center_y = h // 2
        gap_y = int(h * 0.3) # 화면 높이의 20% 간격
        
        for i in range(3):
            fish_y = center_y + ((i - 1) * gap_y) - (food_h // 2)
            foods.append({
                'x': l_margin, 
                'y': fish_y, 
                'orig_x': l_margin,
                'orig_y': fish_y,
                'img': img_fish
            })
        
        initialized = True
        print(f"화면 크기 감지: {w}x{h}")
        print(f"고양이 위치 설정: ({cat_x}, {cat_y})")

    # --- 고양이 상태 ---
    if cat_state == "happy":
        if time.time() - last_fed_time > 1.0:
            cat_state = "hungry"

    current_cat_img = img_cat_happy if cat_state == "happy" else img_cat_hungry
    # 계산된 cat_x, cat_y 사용
    img = overlay_transparent(img, current_cat_img, cat_x, cat_y)

    # --- 손 인식 ---
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    results = hands.detect_for_video(mp_image, timestamper.next())

    hand_cx, hand_cy = 0, 0
    fist_detected = False

    if results.hand_landmarks:
        # Tasks API 결과는 손마다 NormalizedLandmark 리스트를 바로 담고 있습니다.
        for hand_landmarks in results.hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            
            # 손바닥 중심
            hand_cx, hand_cy = lm_list[9][1], lm_list[9][2]

            if is_fist(lm_list):
                fist_detected = True
                cv2.circle(img, (hand_cx, hand_cy), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "GRAB", (hand_cx-20, hand_cy-20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            else:
                cv2.circle(img, (hand_cx, hand_cy), 10, (0, 0, 255), cv2.FILLED)
            break 

    # --- 이동 로직 ---
    # 잡기
    if fist_detected and dragged_idx == -1:
        for i, food in enumerate(foods):
            if (food['x'] < hand_cx < food['x'] + food_w) and \
               (food['y'] < hand_cy < food['y'] + food_h):
                dragged_idx = i
                break

    # 드래그
    if fist_detected and dragged_idx != -1:
        foods[dragged_idx]['x'] = hand_cx - (food_w // 2)
        foods[dragged_idx]['y'] = hand_cy - (food_h // 2)

        # 고양이 충돌 체크
        cat_center_x = cat_x + (cat_w // 2)
        cat_center_y = cat_y + (cat_h // 2)
        
        dist = math.hypot(cat_center_x - hand_cx, cat_center_y - hand_cy)

        if dist < feed_distance:
            score += 1
            cat_state = "happy"
            last_fed_time = time.time()
            foods[dragged_idx]['x'] = foods[dragged_idx]['orig_x']
            foods[dragged_idx]['y'] = foods[dragged_idx]['orig_y']
            dragged_idx = -1

    if not fist_detected:
        dragged_idx = -1

    # --- 생선 그리기 ---
    for food in foods:
        img = overlay_transparent(img, food['img'], int(food['x']), int(food['y']))

    # --- UI ---
    # 수정 코드: (w - 250, 80) -> 전체 너비에서 오른쪽 여백 250px 뺌
    cv2.putText(img, f"Score: {score}", (w - 200, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
    if cat_state == "happy":
        cv2.putText(img, "Yum Yum!", (cat_x, cat_y-20), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    
    # 해상도 정보 표시 (디버깅용)
    cv2.putText(img, f"Res: {w}x{h}", (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Cat Feeding Game", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
