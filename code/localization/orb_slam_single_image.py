import cv2
import os
import numpy as np
import time

import global_config


def load_camera_params(yaml_path):
    """Parses focal length and optical center."""
    if not os.path.exists(yaml_path):
        print("Config file not found. Using defaults.")
        return 718.856, 718.856, 607.1928, 185.2157

    with open(yaml_path, 'r') as f:
        try:
            # Simple parsing
            for line in f:
                if "Camera.fx:" in line:
                    fx = float(line.split(":")[1])
                if "Camera.fy:" in line:
                    fy = float(line.split(":")[1])
                if "Camera.cx:" in line:
                    cx = float(line.split(":")[1])
                if "Camera.cy:" in line:
                    cy = float(line.split(":")[1])
            return fx, fy, cx, cy
        except:
            return 500.0, 500.0, 320.0, 240.0


class VisualOdometryORB:
    def __init__(self, focal_len, pp):
        self.focal_len = focal_len
        self.pp = pp  # Principal Point (cx, cy)

        # 1. Initialize ORB
        # nfeatures=3000 ensures we find enough points even in low texture
        self.orb = cv2.ORB_create(nfeatures=3000)

        # 2. Initialize Matcher
        # Switch to standard BFMatcher (crossCheck=False allows kNN match)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.trajectory = []
        
        # Performance tracking
        self.inference_times = []
        self.max_history = 30
        self.frame_count = 0

    def process_frame(self, image):
        """Process a frame and estimate camera motion."""
        start_time = time.time()
        self.frame_count += 1

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 3. Detect ORB Features & Descriptors
        kp, des = self.orb.detectAndCompute(gray, None)

        # Convert keypoints to numpy array of points for visualization/calculation
        if kp:
            kp_pts = np.float32([k.pt for k in kp])
        else:
            return None  # No features found

        # Initialization (First Frame)
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            # Return current position (0,0,0) and points to visualize
            elapsed = time.time() - start_time
            self.inference_times.append(elapsed)
            return {
                'position': self.cur_t,
                'rotation': self.cur_R,
                'keypoints': kp_pts,
                'num_features': len(kp),
                'status': 'initialized',
                'inference_time': elapsed
            }

        # 4. Match Features (Previous vs Current)
        if des is None or self.prev_des is None:
            # If we lose features, reset prev_frame to current, so we can recover next frame
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            elapsed = time.time() - start_time
            self.inference_times.append(elapsed)
            return {
                'position': self.cur_t,
                'rotation': self.cur_R,
                'keypoints': kp_pts,
                'num_features': len(kp),
                'status': 'feature_lost',
                'inference_time': elapsed
            }

        # Use KNN matching with k=2
        matches = self.matcher.knnMatch(self.prev_des, des, k=2)

        # Apply Lowe's Ratio Test (Robust filtering)
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass  # Handle cases where k<2

        if len(good_matches) < 10:
            # Update history even if tracking failed for this frame
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            # Not enough matches to calculate motion
            elapsed = time.time() - start_time
            self.inference_times.append(elapsed)
            return {
                'position': self.cur_t,
                'rotation': self.cur_R,
                'keypoints': kp_pts,
                'num_features': len(kp),
                'status': 'insufficient_matches',
                'matching_score': 0.0,
                'inference_time': elapsed
            }

        # Extract (x, y) coordinates from the matches
        # Note: queryIdx refers to prev_kp, trainIdx refers to current kp
        # We must perform this BEFORE updating self.prev_kp
        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good_matches])

        # 5. Estimate Motion (Essential Matrix)
        E, mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            self.focal_len,
            self.pp,
            cv2.RANSAC,
            0.999,
            1.0,
            None
        )

        # Recover Rotation (R) and Translation (t)
        if E is not None:
            _, R, t, mask = cv2.recoverPose(E, pts_curr, pts_prev, focal=self.focal_len, pp=self.pp)

            # 6. Update Trajectory
            # Monocular Scale Ambiguity: We assume scale=1.0 because we have no IMU or Wheel Encoders
            absolute_scale = 1.0

            # Only update if the move is "significant" (filters jitter)
            if np.mean(np.abs(t)) > 0.005:
                self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
                self.cur_R = self.cur_R.dot(R)

            # Add to path for drawing
            # Fix DeprecationWarning by extracting scalar value first using float() or [0]
            x = int(float(self.cur_t[0])) + 400
            z = int(float(self.cur_t[2])) + 400
            self.trajectory.append((x, z))

        # 7. Update previous state AFTER all calculations
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des

        # Calculate matching quality score
        matching_score = len(good_matches) / max(len(self.prev_kp), len(kp)) if (self.prev_kp and kp) else 0.0

        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)

        # Return comprehensive localization data
        return {
            'position': self.cur_t.copy(),
            'rotation': self.cur_R.copy(),
            'keypoints': kp_pts,
            'num_features': len(kp),
            'num_matches': len(good_matches),
            'matching_score': matching_score,
            'status': 'tracking',
            'inference_time': elapsed
        }

    def draw_trajectory(self, traj_img_size=(800, 800)):
        traj_img = np.zeros((traj_img_size[0], traj_img_size[1], 3), dtype=np.uint8)

        for i in range(1, len(self.trajectory)):
            cv2.line(traj_img, self.trajectory[i - 1], self.trajectory[i], (0, 255, 0), 2)

        if self.trajectory:
            cv2.circle(traj_img, self.trajectory[-1], 5, (0, 0, 255), -1)
            cv2.putText(traj_img, f"Pos: {self.trajectory[-1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)

        return traj_img

    def get_fps(self):
        """Calculate average FPS from recent inference times."""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_trajectory_displacement(self):
        """Get total 3D displacement from origin."""
        return float(np.linalg.norm(self.cur_t))

    def get_localization_status(self):
        """Get current localization status and statistics."""
        return {
            'trajectory_length': len(self.trajectory),
            'position': self.cur_t.copy(),
            'displacement': self.get_trajectory_displacement(),
            'frame_count': self.frame_count,
            'fps': self.get_fps()
        }


def run_single_image(image_path):
    """Runs the ORB Visual Odometry logic on a single static image."""
    # Load params from your config file
    fx, fy, cx, cy = load_camera_params("webcam_config.yaml")
    vo = VisualOdometryORB(focal_len=fx, pp=(cx, cy))

    print(f"--- Processing Single Image: {image_path} ---")

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read image.")
        return

    # Process the single frame
    # This triggers the "Initialization" block in the class
    result = vo.process_frame(frame)

    display_frame = frame.copy()

    if result is not None:
        keypoints = result['keypoints']
        # Draw features
        for pt in keypoints:
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        status_text = f"ORB Features: {len(keypoints)}"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"Success! Detected {len(keypoints)} features.")
        print("Visual Odometry initialized (Ready for next frame).")
    else:
        print("No features detected.")

    # Show result
    cv2.imshow("ORB Features", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    cv2.imwrite("results/orb_result.jpg", display_frame)


def run_orb_vo_live_enhanced():
    """Enhanced real-time ORB Visual Odometry with better diagnostics."""
    # Load params from your config file
    fx, fy, cx, cy = load_camera_params("webcam_config.yaml")
    vo = VisualOdometryORB(focal_len=fx, pp=(cx, cy))

    cap = cv2.VideoCapture(0)

    print("--- ORB Visual Odometry Started ---")
    print("Green Dots = ORB Features")
    print("Move the camera slowly!")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        result = vo.process_frame(frame)

        display_frame = frame.copy()

        # Draw ORB Features on the camera feed
        if result is not None:
            keypoints = result['keypoints']
            # Draw all features found (Green dots)
            for pt in keypoints:
                cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

            # Add diagnostics
            status_text = f"Status: {result['status']}"
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            features_text = f"Features: {result['num_features']}"
            cv2.putText(display_frame, features_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if 'num_matches' in result:
                matches_text = f"Matches: {result['num_matches']}"
                cv2.putText(display_frame, matches_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            fps_text = f"FPS: {vo.get_fps():.1f}"
            cv2.putText(display_frame, fps_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Finding Features...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        traj_map = vo.draw_trajectory()

        cv2.imshow("ORB Tracking", display_frame)
        cv2.imshow("Trajectory", traj_map)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nSession complete: Processed {frame_count} frames")


if __name__ == "__main__":
    # Replace with your actual image path
    image_path = global_config.image_path

    run_single_image(image_path)
