import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def count_fingers(landmarks):
    fingers = []
    if landmarks[0].x < landmarks[17].x:
        fingers.append(landmarks[4].x < landmarks[3].x)
    else:
        fingers.append(landmarks[4].x > landmarks[3].x)
    fingers.append(landmarks[8].y < landmarks[6].y)
    fingers.append(landmarks[12].y < landmarks[10].y)
    fingers.append(landmarks[16].y < landmarks[14].y)
    fingers.append(landmarks[20].y < landmarks[18].y)
    return sum(fingers)

def get_point(landmarks, idx):
    return np.array([landmarks[idx].x, landmarks[idx].y])

def classify_expression(landmarks):
    def dist(a, b):
        return np.linalg.norm(a - b)
    def get(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y])

    left_mouth = get(61)
    right_mouth = get(291)
    top_lip = get(13)
    bottom_lip = get(14)
    left_eye_upper = get(159)
    left_eye_lower = get(145)
    right_eye_upper = get(386)
    right_eye_lower = get(374)
    left_eyebrow_inner = get(105)
    right_eyebrow_inner = get(334)
    eye_outer_left = get(33)
    eye_outer_right = get(133)

    mouth_width = dist(left_mouth, right_mouth)
    mouth_height = dist(top_lip, bottom_lip)
    mouth_aspect = mouth_height / (mouth_width + 1e-6)

    left_eye_open = dist(left_eye_upper, left_eye_lower)
    right_eye_open = dist(right_eye_upper, right_eye_lower)
    eye_open_avg = (left_eye_open + right_eye_open) / 2

    eye_width = dist(eye_outer_left, eye_outer_right)
    eye_ratio = eye_open_avg / (eye_width + 1e-6)

    brow_raise = ((left_eyebrow_inner[1] + right_eyebrow_inner[1]) / 2) - ((left_eye_upper[1] + right_eye_upper[1]) / 2)

    smile_ratio = mouth_width / (eye_width + 1e-6)
    mouth_drop = bottom_lip[1] - top_lip[1]

    # Expression
    if smile_ratio > 1.8 and mouth_aspect > 0.10:
        return "Happy"
    elif brow_raise > 0.035 and mouth_aspect < 0.13 and mouth_drop < 0.015:
        return "Sad"
    elif mouth_aspect > 0.4 and eye_ratio > 0.32 and brow_raise > 0.01:
        return "Surprised"
    elif mouth_aspect < 0.25 and brow_raise < -0.025 and eye_ratio < 0.22:
        return "Angry"
    else:
        return "Neutral"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera 0")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    expression = "No face"
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1)
        )
        expression = classify_expression(face_landmarks.landmark)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            label = handedness.classification[0].label
            landmarks = hand_landmarks.landmark
            finger_count = count_fingers(landmarks)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            cx = int(landmarks[0].x * w)
            cy = int(landmarks[0].y * h)
            cv2.putText(frame, f"{label} hand: {finger_count}", (cx - 50, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Expression: {expression}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Hand + Face + Expression", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()