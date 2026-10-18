import cv2
import mediapipe as mp
import numpy as np

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def compute_all_metrics(landmarks, img_w, img_h):
    """Compute all LegAlign metrics from MediaPipe landmarks."""
    
    def px(idx):
        return np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h])
    
    # Key points
    l_hip, r_hip = px(23), px(24)
    l_knee, r_knee = px(25), px(26)
    l_ankle, r_ankle = px(27), px(28)
    l_shoulder, r_shoulder = px(11), px(12)
    
    # 1. Leg cosine similarity
    left_leg = l_ankle - l_hip
    right_leg = r_ankle - r_hip
    leg_cosine = cosine_similarity(left_leg, right_leg)
    
    # 2. Pelvic tilt (degrees from horizontal)
    pelvis = r_hip - l_hip
    horizontal = np.array([1.0, 0.0])
    pelvic_cos = np.clip(cosine_similarity(pelvis, horizontal), -1, 1)
    pelvic_tilt = abs(np.degrees(np.arccos(pelvic_cos)))
    
    # 3. Spine deviation (midline shoulder vs midline hip offset)
    hip_mid = (l_hip + r_hip) / 2
    shoulder_mid = (l_shoulder + r_shoulder) / 2
    spine_vec = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0])  # pointing up in image coords
    spine_cos = np.clip(cosine_similarity(spine_vec, vertical), -1, 1)
    spine_deviation = abs(np.degrees(np.arccos(spine_cos)))
    
    # 4. Knee flexion angles (hip-knee-ankle angle)
    def knee_angle(hip, knee, ankle):
        v1 = hip - knee
        v2 = ankle - knee
        cos_val = np.clip(cosine_similarity(v1, v2), -1, 1)
        return np.degrees(np.arccos(cos_val))
    
    left_knee_angle = knee_angle(l_hip, l_knee, l_ankle)
    right_knee_angle = knee_angle(r_hip, r_knee, r_ankle)
    
    # 5. Pixel leg lengths
    left_leg_px = np.linalg.norm(left_leg)
    right_leg_px = np.linalg.norm(right_leg)
    
    return {
        'leg_cosine': leg_cosine,
        'pelvic_tilt_deg': pelvic_tilt,
        'spine_deviation_deg': spine_deviation,
        'left_knee_angle': left_knee_angle,
        'right_knee_angle': right_knee_angle,
        'left_leg_px': left_leg_px,
        'right_leg_px': right_leg_px,
        'leg_length_diff_px': abs(left_leg_px - right_leg_px)
    }

def compute_legalign_score(metrics):
    """
    Composite LegAlign Score (0-100).
    0 = no discrepancy detected
    100 = maximum discrepancy
    
    Weights: leg_cosine 50%, pelvic_tilt 35%, spine 15%
    """
    # Convert leg cosine to a 0-100 deficit score
    # 1.000 → 0 (perfect), 0.950 → 100 (severe)
    cos_score = min(100, max(0, (1.0 - metrics['leg_cosine']) / 0.05 * 100))
    
    # Convert pelvic tilt: 0° → 0, 7°+ → 100
    tilt_score = min(100, max(0, metrics['pelvic_tilt_deg'] / 7.0 * 100))
    
    # Convert spine deviation: 0° → 0, 15°+ → 100
    spine_score = min(100, max(0, metrics['spine_deviation_deg'] / 15.0 * 100))
    
    # Weighted composite
    composite = (cos_score * 0.50) + (tilt_score * 0.35) + (spine_score * 0.15)
    
    return round(composite, 1)

def classify_risk(score):
    """Classify LegAlign score into risk levels."""
    if score < 20:
        return "Normal", "No significant discrepancy detected"
    elif score < 40:
        return "Mild", "Minor asymmetry — monitor over time"
    elif score < 60:
        return "Moderate", "Recommend specialist consultation"
    else:
        return "Significant", "Strong indicators of LLD — seek evaluation"

# === MAIN ===
image = cv2.imread('test_person.jpg')
h, w = image.shape[:2]

mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if not results.pose_landmarks:
    print("No pose detected!")
    exit()

metrics = compute_all_metrics(results.pose_landmarks.landmark, w, h)
score = compute_legalign_score(metrics)
risk_level, description = classify_risk(score)

print("=" * 55)
print("         LegAlign Full Analysis Report")
print("=" * 55)
print(f"\n  LEGALIGN SCORE: {score}/100  [{risk_level}]")
print(f"  {description}")
print(f"\n  --- Detailed Metrics ---")
print(f"  Leg Cosine Similarity:  {metrics['leg_cosine']:.6f}")
print(f"  Pelvic Tilt:            {metrics['pelvic_tilt_deg']:.2f}°")
print(f"  Spine Deviation:        {metrics['spine_deviation_deg']:.2f}°")
print(f"  Left Knee Angle:        {metrics['left_knee_angle']:.1f}°")
print(f"  Right Knee Angle:       {metrics['right_knee_angle']:.1f}°")
print(f"  Left Leg (pixels):      {metrics['left_leg_px']:.1f}")
print(f"  Right Leg (pixels):     {metrics['right_leg_px']:.1f}")
print(f"  Pixel Difference:       {metrics['leg_length_diff_px']:.1f} px")
print("=" * 55)