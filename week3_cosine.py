import os
import cv2
import mediapipe as mp
import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)

def pelvic_tilt_degrees(left_hip, right_hip):
    pelvis_vector = right_hip - left_hip
    horizontal = np.array([1.0, 0.0])
    
    cos_val = cosine_similarity(pelvis_vector, horizontal)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_val))
    
    return abs(angle)

# === 1. LOAD IMAGE & CHECK ===
image_path = 'test_person.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"ERROR: Could not find '{image_path}'. Make sure it is in this folder:")
    print(f" → {os.getcwd()}")
    exit()

h, w = image.shape[:2]

# === 2. PROCESS MEDIAPIPE ===
mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if not results.pose_landmarks:
    print("ERROR: MediaPipe did not detect a person in the image.")
    exit()

lm = results.pose_landmarks.landmark

# Get pixel coordinates
def px(idx):
    return np.array([lm[idx].x * w, lm[idx].y * h])

left_hip, right_hip = px(23), px(24)
left_ankle, right_ankle = px(27), px(28)

# Leg vectors
left_leg = left_ankle - left_hip
right_leg = right_ankle - right_hip

# === 3. CORE METRICS ===
leg_cosine = cosine_similarity(left_leg, right_leg)
tilt = pelvic_tilt_degrees(left_hip, right_hip)

print("=" * 50)
print("       LegAlign Core Metrics")
print("=" * 50)
print(f"  Leg Cosine Similarity: {leg_cosine:.6f}")
print(f"  Pelvic Tilt:           {tilt:.2f} degrees")
print()

if leg_cosine >= 0.985:
    print("  → Legs appear well-aligned (Normal)")
elif leg_cosine >= 0.970:
    print("  → Mild asymmetry detected (Monitor)")
else:
    print("  → Significant asymmetry (Recommend evaluation)")

if tilt < 1.5:
    print(f"  → Pelvis appears level (tilt = {tilt:.1f}°)")
elif tilt < 3.0:
    print(f"  → Mild pelvic tilt detected ({tilt:.1f}°)")
else:
    print(f"  → Significant pelvic tilt ({tilt:.1f}° — LLD suspected)")

# === 4. DRAW & SAVE VISUALIZATION ===
annotated = image.copy()

# Draw Pelvis Baseline Line (Cyan)
cv2.line(annotated, tuple(left_hip.astype(int)), tuple(right_hip.astype(int)), (255, 255, 0), 4)

# Draw Left Leg Vector Line (Green)
cv2.line(annotated, tuple(left_hip.astype(int)), tuple(left_ankle.astype(int)), (0, 255, 0), 4)

# Draw Right Leg Vector Line (Red)
cv2.line(annotated, tuple(right_hip.astype(int)), tuple(right_ankle.astype(int)), (0, 0, 255), 4)

# Save image
output_filename = 'week2_vectors.jpg'
success = cv2.imwrite(output_filename, annotated)

if success:
    print(f"\nSUCCESS: Output map saved to {os.path.join(os.getcwd(), output_filename)}")
else:
    print("\nERROR: cv2.imwrite failed to save the image file.")