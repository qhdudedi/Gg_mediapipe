import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_styles,
    drawing_utils,
)

from mp_utils import Timestamper, ensure_model

# MediaPipe 1.x 부터는 mp.solutions.hands 가 없어지고 Tasks API 를 사용합니다.
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
    running_mode=RunningMode.VIDEO,
    num_hands=2,                        # 최대 인식할 손의 개수
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 웹캠 설정
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    timestamper = Timestamper()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # 좌우 반전 후 RGB 변환 (MediaPipe 입력은 RGB)
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 랜드마크 추적 수행 (VIDEO 모드는 증가하는 타임스탬프가 필요)
        result = landmarker.detect_for_video(mp_image, timestamper.next())

        for hand_landmarks in result.hand_landmarks:
            # 랜드마크 그리기
            drawing_utils.draw_landmarks(
                image,
                hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키로 종료
            break

cap.release()
cv2.destroyAllWindows()
