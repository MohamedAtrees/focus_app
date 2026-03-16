import cv2
import pygame
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque

PHONE_CONF = 0.35
PHONE_CONFIRM_FRAMES = 5
ALERT_COOLDOWN = 3.0
YAW_THRESHOLD = 25
PITCH_THRESHOLD = 20

pygame.mixer.init()
try:
    annoying_sound = pygame.mixer.Sound(r"C:\Users\matre\Pictures\collage\app\song1.mp3")
except Exception as e:
    print(f"Audio error: {e}")
    exit()

is_playing = False
last_alert_time = 0

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

phone_buffer = deque(maxlen=PHONE_CONFIRM_FRAMES)

def get_head_pose(landmarks, frame_w, frame_h):
    face_3d = np.array([
        [0.0,      0.0,      0.0   ],
        [0.0,     -330.0,   -65.0  ],
        [-225.0,   170.0,   -135.0 ],
        [225.0,    170.0,   -135.0 ],
        [-150.0,  -150.0,  -125.0  ],
        [150.0,   -150.0,  -125.0  ]
    ], dtype=np.float64)

    ids = [1, 152, 263, 33, 287, 57]
    face_2d = []
    for idx in ids:
        lm = landmarks[idx]
        face_2d.append([lm.x * frame_w, lm.y * frame_h])
    face_2d = np.array(face_2d, dtype=np.float64)

    focal = frame_w
    cam_matrix = np.array([
        [focal, 0,     frame_w / 2],
        [0,     focal, frame_h / 2],
        [0,     0,     1          ]
    ], dtype=np.float64)

    dist = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist)
    if not success:
        return 0, 0, 0

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

    pitch = angles[0] * 360
    yaw   = angles[1] * 360
    roll  = angles[2] * 360

    return pitch, yaw, roll


print("Focus Monitor Started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    results = model(frame, stream=True, verbose=False, conf=PHONE_CONF)
    phone_this_frame = False
    phone_box = None

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 67:
                phone_this_frame = True
                phone_box = list(map(int, box.xyxy[0]))
                break

    phone_buffer.append(phone_this_frame)
    phone_confirmed = phone_buffer.count(True) >= (PHONE_CONFIRM_FRAMES * 0.6)

    if phone_box:
        x1, y1, x2, y2 = phone_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 50, 50), 2)
        cv2.putText(frame, "Phone", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_results = face_mesh.process(rgb)

    pitch, yaw, roll = 0, 0, 0
    face_detected = False
    looking_at_phone = False

    if mesh_results.multi_face_landmarks:
        face_detected = True
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        pitch, yaw, roll = get_head_pose(landmarks, w, h)

        looking_at_phone = (
            abs(yaw) > YAW_THRESHOLD or
            abs(pitch) > PITCH_THRESHOLD
        )

        nose = landmarks[1]
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        arrow_x = int(nose_x + yaw * 2)
        arrow_y = int(nose_y - pitch * 2)
        cv2.arrowedLine(frame, (nose_x, nose_y), (arrow_x, arrow_y),
                        (0, 255, 255), 2, tipLength=0.3)

    now = time.time()
    should_alert = phone_confirmed and looking_at_phone

    if should_alert:
        if not is_playing and (now - last_alert_time) > ALERT_COOLDOWN:
            annoying_sound.play(loops=-1)
            is_playing = True
            last_alert_time = now
            print("ALERT: Looking at phone!")
    else:
        if is_playing:
            annoying_sound.stop()
            is_playing = False

    status_color = (0, 0, 255) if should_alert else (0, 200, 0)
    cv2.putText(frame, "ALERT: Focus!" if should_alert else "OK",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

    cv2.putText(frame, f"Phone: {'YES' if phone_confirmed else 'no'}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 100, 0) if phone_confirmed else (150, 150, 150), 2)

    if face_detected:
        gaze_color = (0, 0, 255) if looking_at_phone else (0, 200, 0)
        cv2.putText(frame, f"Gaze: {'PHONE' if looking_at_phone else 'OK'} (Y:{yaw:.0f} P:{pitch:.0f})",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gaze_color, 2)
    else:
        cv2.putText(frame, "No face detected",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 2)

    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Focus Monitor", small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Focus Monitor", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
pygame.quit()