from pathlib import Path

import cv2
import mediapipe as mp


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_DIR / "test_person.jpg"
MODEL_PATH = PROJECT_DIR / "pose_landmarker_heavy.task"
OUTPUT_PATH = PROJECT_DIR / "week1_output_skeleton.jpg"

if not MODEL_PATH.is_file():
    raise SystemExit(
        f"ERROR: Model not found: {MODEL_PATH}\n"
        "Download the Pose landmarker (Heavy) .task model and save it there."
    )

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise SystemExit(f"ERROR: Could not load image: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
)

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
    results = landmarker.detect(mp_image)

if not results.pose_landmarks:
    raise SystemExit("ERROR: No person detected in the image")

# The first detected person's landmarks, indexed 0–32.
landmarks = results.pose_landmarks[0]

print("=== All 33 MediaPipe Landmarks ===")
for idx, landmark in enumerate(landmarks):
    print(
        f"  [{idx:2d}] x={landmark.x:.4f}, "
        f"y={landmark.y:.4f}, z={landmark.z:.4f}, "
        f"visibility={landmark.visibility:.2f}"
    )

important = {
    11: "Left Shoulder", 12: "Right Shoulder",
    23: "Left Hip",      24: "Right Hip",
    25: "Left Knee",     26: "Right Knee",
    27: "Left Ankle",    28: "Right Ankle",
}

print("\n=== Key Landmarks for LegAlign ===")
for idx, name in important.items():
    lm = landmarks[idx]
    print(
        f"  {name:15s} → x={lm.x:.4f}, "
        f"y={lm.y:.4f}, visibility={lm.visibility:.2f}"
    )

annotated = image.copy()
height, width = image.shape[:2]


def pixel_point(landmark):
    return (
        int(landmark.x * width),
        int(landmark.y * height),
    )


# Draw the Tasks API's pose connections instead of mp.solutions.drawing_utils.
for connection in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS:
    start = landmarks[connection.start]
    end = landmarks[connection.end]
    if start.visibility > 0.5 and end.visibility > 0.5:
        cv2.line(
            annotated,
            pixel_point(start),
            pixel_point(end),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

for idx, landmark in enumerate(landmarks):
    if landmark.visibility <= 0.5:
        continue

    cx, cy = pixel_point(landmark)
    cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1, cv2.LINE_AA)

    text_x = cx + 30
    text_y = cy - 20

    cv2.arrowedLine(
        annotated, (text_x, text_y), (cx, cy),
        (0, 0, 0), 4, tipLength=0.2
    )
    cv2.arrowedLine(
        annotated, (text_x, text_y), (cx, cy),
        (255, 255, 0), 2, tipLength=0.2
    )
    cv2.putText(
        annotated, str(idx), (text_x + 5, text_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        annotated, str(idx), (text_x + 5, text_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 255), 2, cv2.LINE_AA
    )

if not cv2.imwrite(str(OUTPUT_PATH), annotated):
    raise SystemExit(f"ERROR: Could not save image: {OUTPUT_PATH}")

print(f"\nSaved annotated image with arrows and numbers → {OUTPUT_PATH}")