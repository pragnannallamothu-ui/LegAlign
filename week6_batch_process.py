import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from week4_full_pipeline import compute_all_metrics, compute_legalign_score, classify_risk

def batch_analyze(image_folder, output_csv='results/batch_results.csv'):
    """Process all images in a folder, output CSV."""
    
    mp_pose = mp.solutions.pose
    results_data = []
    
    # Get all jpg files
    images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')])
    print(f"Found {len(images)} images to process...")
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        for i, filename in enumerate(images):
            filepath = os.path.join(image_folder, filename)
            image = cv2.imread(filepath)
            
            if image is None:
                print(f"  SKIP: {filename} (unreadable)")
                continue
            
            h, w = image.shape[:2]
            result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not result.pose_landmarks:
                print(f"  SKIP: {filename} (no pose detected)")
                continue
            
            metrics = compute_all_metrics(result.pose_landmarks.landmark, w, h)
            score = compute_legalign_score(metrics)
            risk, _ = classify_risk(score)
            
            # Parse filename: S01_2cm_front_rep1.jpg
            parts = filename.replace('.jpg', '').split('_')
            subject_id = parts[0] if len(parts) >= 1 else ''
            condition = parts[1] if len(parts) >= 2 else ''
            view = parts[2] if len(parts) >= 3 else ''
            rep = parts[3] if len(parts) >= 4 else ''
            
            row = {
                'filename': filename,
                'subject_id': subject_id,
                'condition_cm': condition,
                'view': view,
                'rep': rep,
                'leg_cosine': f"{metrics['leg_cosine']:.6f}",
                'pelvic_tilt_deg': f"{metrics['pelvic_tilt_deg']:.2f}",
                'spine_deviation_deg': f"{metrics['spine_deviation_deg']:.2f}",
                'leg_diff_px': f"{metrics['leg_length_diff_px']:.1f}",
                'legalign_score': score,
                'risk_level': risk
            }
            results_data.append(row)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(images)}...")
    
    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_data[0].keys())
        writer.writeheader()
        writer.writerows(results_data)
    
    print(f"\nDone! Results saved to {output_csv}")
    print(f"Processed: {len(results_data)} images successfully")
    return results_data

if __name__ == '__main__':
    batch_analyze('./')