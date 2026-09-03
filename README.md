# mediapipe 실습
참고 문서 : https://developers.google.com/edge/mediapipe/solutions/guide

# 실습 환경
```
# 가상환경
uv init --bare
```

# 설치 라이브러리
```
uv add mediapipe opencv-python 
uv add dlib-bin
uv pip install face-recognition --no-deps
```

# 주의사항
- MediaPipe 1.x 부터 `mp.solutions.hands / pose / face_mesh` 가 삭제되었음.
- Tasks API(`mediapipe.tasks.python.vision`) 로 변경됨
- Tasks API 는 `.task` 모델 파일이 필요함. 
- `mp_utils.ensure_model()` 이 최초 실행 시 `models/` 폴더로 자동 다운로드됨