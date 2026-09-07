import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load image
image = cv2.imread('test_person.jpg')
if image is None:
    print("ERROR: Could not load test_person.jpg — check the file exists")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run pose detection
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
) as pose:
    results = pose.process(image_rgb)

if not results.pose_landmarks:
    print("ERROR: No person detected in the image")
    exit()

# Print all 33 landmarks to the terminal
print("=== All 33 MediaPipe Landmarks ===")
for idx, landmark in enumerate(results.pose_landmarks.landmark):
    print(f"  [{idx:2d}] x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f}, visibility={landmark.visibility:.2f}")

# Highlight the landmarks WE care about for LLD
important = {
    11: "Left Shoulder", 12: "Right Shoulder",
    23: "Left Hip",      24: "Right Hip",
    25: "Left Knee",     26: "Right Knee",
    27: "Left Ankle",    28: "Right Ankle"
}

print("\n=== Key Landmarks for LegAlign ===")
for idx, name in important.items():
    lm = results.pose_landmarks.landmark[idx]
    print(f"  {name:15s} → x={lm.x:.4f}, y={lm.y:.4f}, visibility={lm.visibility:.2f}")

# Draw skeleton lines on image
annotated = image.copy()
mp_drawing.draw_landmarks(
    annotated,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS
)

# --- High-Visibility Arrows Pointing to Landmarks ---
h, w, c = image.shape  
for idx, landmark in enumerate(results.pose_landmarks.landmark):
    if landmark.visibility > 0.5:  
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        
        # Calculate text position (offsetting the numbers away from the joint)
        text_x = cx + 30
        text_y = cy - 20
        
        # 1. Draw a bold black backing arrow line (Shadow)
        cv2.arrowedLine(annotated, (text_x, text_y), (cx, cy), (0, 0, 0), 4, tipLength=0.2)
        # 2. Draw a bright cyan foreground arrow line
        cv2.arrowedLine(annotated, (text_x, text_y), (cx, cy), (255, 255, 0), 2, tipLength=0.2)
        
        # 3. Draw a bold black outline for the text numbers
        cv2.putText(
            annotated, str(idx), (text_x + 5, text_y + 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA
        )
        # 4. Draw bright yellow text numbers on top
        cv2.putText(
            annotated, str(idx), (text_x + 5, text_y + 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )
# ----------------------------------------------------

# Save the newly numbered image
cv2.imwrite('week1_output_skeleton.jpg', annotated)
print("\nSaved annotated image with arrows and numbers → week1_output_skeleton.jpg")
