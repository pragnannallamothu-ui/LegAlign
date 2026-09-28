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
    """
    Calculates the true deviation of the pelvis line from the horizontal axis.
    Returns an angle in degrees where 0° means a perfectly level pelvis.
    """
    dy = right_hip[1] - left_hip[1]
    dx = right_hip[0] - left_hip[0]
    
    # Calculate angle using arctan2 (-180 to 180 degrees)
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    # Normalize the angle to map any direction vector back to a 0-90 degree deviation scale
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
        
    return abs(angle_deg)

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

# Get precise pixel coordinates
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
print("       LegAlign Core Metrics (Refined)")
print("=" * 50)
print(f"  Leg Cosine Similarity: {leg_cosine:.6f}")
print(f"  Pelvic Tilt:           {tilt:.2f} degrees")
print()

# Leg alignment evaluation
if leg_cosine >= 0.985:
    print("  → Legs appear well-aligned (Normal)")
elif leg_cosine >= 0.970:
    print("  → Mild asymmetry detected (Monitor)")
else:
    print("  → Significant asymmetry (Recommend evaluation)")

# Pelvic tilt evaluation with a realistic normal buffer (< 3.0°)
if tilt < 3.0:
    print(f"  → Pelvis appears level (tilt = {tilt:.2f}° — Normal)")
elif tilt < 5.0:
    print(f"  → Mild pelvic tilt detected ({tilt:.2f}° — Monitor)")
else:
    print(f"  → Significant pelvic tilt ({tilt:.2f}° — LLD suspected)")

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
