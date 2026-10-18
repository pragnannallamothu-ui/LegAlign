import cv2
import numpy as np

def calibrate_ruler(image, ruler_length_cm=30.0):
    """
    Semi-automatic ruler detection.
    For now: manually measure ruler endpoints in the image.
    
    TODO: Later automate with edge detection.
    
    For the exercise, open the image in any image editor,
    note the pixel coordinates of the ruler's left and right ends.
    """
    # MANUAL INPUT (replace with your measured values):
    ruler_left_px = np.array([100, 500])   # ← Change these!
    ruler_right_px = np.array([250, 500])  # ← Change these!
    
    ruler_pixels = np.linalg.norm(ruler_right_px - ruler_left_px)
    pixels_per_cm = ruler_pixels / ruler_length_cm
    
    print(f"  Ruler spans {ruler_pixels:.1f} pixels")
    print(f"  Scale: {pixels_per_cm:.2f} pixels/cm")
    print(f"  1 pixel = {1/pixels_per_cm:.4f} cm")
    
    return pixels_per_cm

def convert_to_cm(metrics, pixels_per_cm):
    """Convert pixel-based measurements to centimeters."""
    return {
        'left_leg_cm': metrics['left_leg_px'] / pixels_per_cm,
        'right_leg_cm': metrics['right_leg_px'] / pixels_per_cm,
        'leg_length_diff_cm': metrics['leg_length_diff_px'] / pixels_per_cm,
        # Angular measurements don't change (already in degrees)
        'pelvic_tilt_deg': metrics['pelvic_tilt_deg'],
        'leg_cosine': metrics['leg_cosine'],
        'spine_deviation_deg': metrics['spine_deviation_deg'],
    }

def analyze_multiple_views(image_paths):
    """
    Process front, back, left, right views and aggregate.
    """
    import mediapipe as mp
    
    all_metrics = {}
    mp_pose = mp.solutions.pose
    
    # Import from week 4
    from week4_full_pipeline import compute_all_metrics, compute_legalign_score
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        for view_name, path in image_paths.items():
            image = cv2.imread(path)
            if image is None:
                print(f"  WARNING: Could not load {path}")
                continue
            
            h, w = image.shape[:2]
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                metrics = compute_all_metrics(results.pose_landmarks.landmark, w, h)
                all_metrics[view_name] = metrics
                print(f"  {view_name:6s}: cosine={metrics['leg_cosine']:.4f}, tilt={metrics['pelvic_tilt_deg']:.2f}°")
            else:
                print(f"  {view_name:6s}: NO POSE DETECTED")
    
    # Aggregate: average the front and back view metrics (most reliable)
    if 'front' in all_metrics and 'back' in all_metrics:
        avg_cosine = (all_metrics['front']['leg_cosine'] + all_metrics['back']['leg_cosine']) / 2
        avg_tilt = (all_metrics['front']['pelvic_tilt_deg'] + all_metrics['back']['pelvic_tilt_deg']) / 2
        print(f"\n  AVERAGED (front+back): cosine={avg_cosine:.4f}, tilt={avg_tilt:.2f}°")
    
    return all_metrics

# Example usage
if __name__ == '__main__':
    print("=== Ruler Calibration ===")
    # Replace with your actual image that has a ruler visible
    # ppc = calibrate_ruler(cv2.imread('front_with_ruler.jpg'))
    
    print("\n=== Multi-View Analysis ===")
    views = {
        'front': 'test_person.jpg',
        #'back':  'images/subject01_back.jpg',
        #'left':  'images/subject01_left.jpg',
        #'right': 'images/subject01_right.jpg',
    }
    # Uncomment when you have real photos:
    analyze_multiple_views(views)
    print("(Replace paths with your actual images to run)")