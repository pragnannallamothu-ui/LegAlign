import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import mediapipe as mp
import numpy as np

def get_landmark_pixel(landmark, image_width, image_height):
    """Convert normalized MediaPipe coords to pixel coords."""
    return np.array([
        landmark.x * image_width,
        landmark.y * image_height
    ])

# Load and process the image from your custom folder
image = cv2.imread('test_person.jpg')
if image is None:
    print("ERROR: Could not load test_person.jpg — check the file name and folder exist")
    exit()

h, w = image.shape[:2]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize Pose Tracker
mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(image_rgb)

if not results.pose_landmarks:
    print("ERROR: No person detected in the image")
    exit()

lm = results.pose_landmarks.landmark

# Extract your key lower-body joints in absolute pixel coordinates
left_hip   = get_landmark_pixel(lm[23], w, h)
right_hip  = get_landmark_pixel(lm[24], w, h)
left_knee  = get_landmark_pixel(lm[25], w, h)
right_knee = get_landmark_pixel(lm[26], w, h)
left_ankle = get_landmark_pixel(lm[27], w, h)
right_ankle= get_landmark_pixel(lm[28], w, h)

# Compute directional leg vectors (Hip starting point → Ankle ending point)
left_leg_vector  = left_ankle - left_hip
right_leg_vector = right_ankle - right_hip

# Calculate vector lengths (magnitudes) using NumPy linear algebra tools
left_leg_length  = np.linalg.norm(left_leg_vector)
right_leg_length = np.linalg.norm(right_leg_vector)
pixel_difference = abs(left_leg_length - right_leg_length)

print("=== Leg Vectors (pixel coordinates) ===")
print(f"  Left leg vector:  {left_leg_vector}")
print(f"  Right leg vector: {right_leg_vector}")
print(f"  Left leg length:  {left_leg_length:.1f} pixels")
print(f"  Right leg length: {right_leg_length:.1f} pixels")
print(f"  Difference:       {pixel_difference:.1f} px")

# Create custom vector visualization map
annotated = image.copy()

# Draw Pelvis Baseline Line (Cyan) between Hips 23 and 24
cv2.line(annotated, tuple(left_hip.astype(int)), tuple(right_hip.astype(int)), (255, 255, 0), 4)

# Draw Left Leg Vector Line (Green) from Hip 23 to Ankle 27
cv2.line(annotated, tuple(left_hip.astype(int)), tuple(left_ankle.astype(int)), (0, 255, 0), 4)

# Draw Right Leg Vector Line (Red) from Hip 24 to Ankle 28
cv2.line(annotated, tuple(right_hip.astype(int)), tuple(right_ankle.astype(int)), (0, 0, 255), 4)

# Save the diagnostic image output
cv2.imwrite('week2_vectors.jpg', annotated)
print("\nSaved output map → week2_vectors.jpg (Green=Left Leg, Red=Right Leg, Cyan=Pelvis)")
