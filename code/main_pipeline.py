"""
Main Integrated Pipeline
=========================
Combines YOLOv12, MiDaS, ORB-SLAM, Contextual Awareness, and Audio Guidance
into a unified real-time assistive navigation system.

"""

import cv2
import torch
import numpy as np
import time
import os
from Yolo12 import YOLOv12Detector
from my_midas import MiDaSDepthEstimator
from orb_slam_single_image import VisualOdometryORB, load_camera_params
from contextual_awareness import ContextualAwarenessEngine
from audio_guidance import AudioGuidanceSystem, SafetyAlerts


class AssistiveNavigationPipeline:
    """
    Main unified pipeline for real-time assistive navigation.
    Integrates all computer vision and AI components.
    """

    def __init__(self, enable_audio=True, enable_display=True, confidence_threshold=0.45):
        """
        Initialize the complete navigation pipeline.

        Args:
            enable_audio (bool): Enable audio guidance
            enable_display (bool): Enable visual display
            confidence_threshold (float): Minimum detection confidence
        """
        print("\n" + "="*70)
        print("ASSISTIVE NAVIGATION SYSTEM - INITIALIZATION")
        print("="*70)

        # Check device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[Pipeline] Device: {self.device}")
        if self.device == 'cuda':
            print(f"[Pipeline] GPU: {torch.cuda.get_device_name(0)}")

        # Initialize components
        print("\n[Pipeline] Initializing components...")
        try:
            self.detector = YOLOv12Detector(
                model_path="yolo12l.pt",
                confidence_threshold=confidence_threshold,
                device=self.device
            )
            print("[Pipeline] YOLOv12 initialized ✓")
        except Exception as e:
            print(f"[Pipeline] ERROR initializing YOLOv12: {e}")
            raise

        try:
            self.depth_estimator = MiDaSDepthEstimator(device=self.device)
            print("[Pipeline] MiDaS initialized ✓")
        except Exception as e:
            print(f"[Pipeline] ERROR initializing MiDaS: {e}")
            raise

        try:
            fx, fy, cx, cy = load_camera_params("webcam_config.yaml")
            self.vo = VisualOdometryORB(focal_len=fx, pp=(cx, cy))
            print("[Pipeline] ORB-SLAM initialized ✓")
        except Exception as e:
            print(f"[Pipeline] ERROR initializing ORB-SLAM: {e}")
            raise

        try:
            self.awareness = ContextualAwarenessEngine(frame_width=640, frame_height=480)
            print("[Pipeline] Contextual Awareness initialized ✓")
        except Exception as e:
            print(f"[Pipeline] ERROR initializing Awareness Engine: {e}")
            raise

        if enable_audio:
            try:
                self.audio = AudioGuidanceSystem(rate=150, volume=1.0)
                print("[Pipeline] Audio Guidance initialized ✓")
                self.enable_audio = True
            except Exception as e:
                print(f"[Pipeline] WARNING: Could not initialize audio: {e}")
                self.audio = None
                self.enable_audio = False
        else:
            self.audio = None
            self.enable_audio = False

        self.enable_display = enable_display

        # Statistics
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        
        print("\n[Pipeline] Initialization complete! System ready for real-time navigation.")
        print("="*70 + "\n")

    def process_frame(self, frame):
        """
        Process a single frame through the complete pipeline.

        Args:
            frame (np.ndarray): Input frame from camera (BGR)

        Returns:
            dict: Comprehensive processing results
        """
        # Validate frame
        if frame is None or not isinstance(frame, np.ndarray):
            print(f"[Pipeline] ERROR: Invalid frame - expected numpy array, got {type(frame)}")
            return {
                'status': 'error: invalid_frame',
                'detections': [],
                'depth_map': None,
                'localization': None,
                'alerts': [],
                'guidance': '',
                'frame_id': self.frame_count + 1
            }
        
        frame_start = time.time()
        self.frame_count += 1

        result = {
            'frame_id': self.frame_count,
            'frame_shape': frame.shape,
            'detections': [],
            'depth_map': None,
            'localization': None,
            'context_analysis': None,
            'alerts': [],
            'guidance': '',
            'timings': {},
            'status': 'processing'
        }

        try:
            # ============================================
            # STEP 1: Object Detection (YOLOv12)
            # ============================================
            yolo_start = time.time()
            detections = self.detector.detect_objects(frame, return_annotated=False)
            result['yolo_time'] = time.time() - yolo_start
            result['detections'] = detections

            # ============================================
            # STEP 2: Depth Estimation (MiDaS)
            # ============================================
            midas_start = time.time()
            depth_map = self.depth_estimator.estimate_depth(frame)
            result['midas_time'] = time.time() - midas_start
            result['depth_map'] = depth_map

            # ============================================
            # STEP 3: Fuse YOLO + MiDaS
            # ============================================
            if depth_map is not None and detections:
                detections_with_depth = self.depth_estimator.get_distances_for_detections(
                    depth_map, detections
                )
                result['detections'] = detections_with_depth

            # ============================================
            # STEP 4: Visual Localization (ORB-SLAM)
            # ============================================
            vo_start = time.time()
            localization = self.vo.process_frame(frame)
            result['vo_time'] = time.time() - vo_start
            result['localization'] = localization

            # ============================================
            # STEP 5: Contextual Analysis
            # ============================================
            context_start = time.time()
            context = self.awareness.analyze_detections(
                result['detections'],
                frame_shape=frame.shape
            )
            result['context_analysis'] = context
            result['context_time'] = time.time() - context_start

            # Extract alerts and guidance
            result['alerts'] = context.get('alerts', [])
            result['guidance'] = context.get('navigation_guidance', '')
            result['safety_score'] = context.get('overall_safety_score', 10.0)

            # ============================================
            # STEP 6: Audio Feedback
            # ============================================
            if self.enable_audio and self.audio and result['alerts']:
                # Speak the top alert
                top_alert = result['alerts'][0]
                audio_start = time.time()
                self.audio.speak(top_alert['message'], priority=top_alert['priority'], wait=False)
                result['audio_time'] = time.time() - audio_start

            result['status'] = 'success'

        except Exception as e:
            print(f"[Pipeline] ERROR processing frame: {str(e)}")
            result['status'] = f'error: {str(e)}'

        # ============================================
        # Calculate total inference time
        # ============================================
        result['total_time'] = time.time() - frame_start
        self.total_inference_time += result['total_time']

        return result

    def create_visualization(self, frame, result):
        """
        Create visualization with detections, depth, and status info.

        Args:
            frame (np.ndarray): Original input frame
            result (dict): Processing result from process_frame()

        Returns:
            np.ndarray: Annotated visualization frame
        """
        # Validate frame
        if frame is None or not isinstance(frame, np.ndarray):
            print(f"[Pipeline] WARNING: Invalid frame type: {type(frame)}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            vis_frame = frame.copy()
        except Exception as e:
            print(f"[Pipeline] WARNING: Could not copy frame: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        h, w = vis_frame.shape[:2]

        # Draw detections
        for det in result['detections']:
            bbox = det.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            
            # Color based on distance
            distance = det.get('distance', 999.0)
            if distance < 1.0:
                color = (0, 0, 255)  # Red - critical
            elif distance < 2.5:
                color = (0, 165, 255)  # Orange - warning
            else:
                color = (0, 255, 0)  # Green - safe

            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Label with distance
            label = f"{det['class']}: {det['confidence']:.2f}"
            if distance < 999.0:
                label += f" ({distance:.1f}m)"

            cv2.putText(
                vis_frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Draw ORB features
        if result['localization'] and 'keypoints' in result['localization']:
            keypoints = result['localization']['keypoints']
            for pt in keypoints:
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)

        # Draw status panel (top-left)
        panel_start_y = 20
        cv2.putText(vis_frame, f"Frame: {result['frame_id']}", (10, panel_start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        fps = self.get_fps()
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, panel_start_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detection stats
        num_detections = len(result['detections'])
        cv2.putText(vis_frame, f"Objects: {num_detections}", (10, panel_start_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Safety score
        safety = result.get('safety_score', 10.0)
        safety_color = (0, 255, 0) if safety >= 7 else (0, 165, 255) if safety >= 4 else (0, 0, 255)
        cv2.putText(vis_frame, f"Safety: {safety:.1f}/10", (10, panel_start_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, safety_color, 2)

        # Guidance text (bottom)
        guidance = result.get('guidance', 'Path is clear')
        cv2.putText(vis_frame, f"Guidance: {guidance}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Alerts (top-right)
        alerts = result.get('alerts', [])
        for i, alert in enumerate(alerts[:3]):  # Show top 3 alerts
            alert_color = (0, 0, 255) if alert['level'] == 'CRITICAL' else (0, 165, 255)
            cv2.putText(vis_frame, f"{alert['level']}: {alert['message'][:40]}", 
                        (w - 400, 20 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)

        return vis_frame

    # ======================== WEBCAM MODE (COMMENTED OUT) ========================
    # def run_real_time(self, camera_id=0, max_frames=None):
    #     """
    #     Run the pipeline in real-time mode on webcam.
    #     DEPRECATED: System now uses image-based processing
    #
    #     Args:
    #         camera_id (int): Camera device ID (default 0)
    #         max_frames (int): Maximum frames to process (None = infinite)
    #     """
    #     print("\n[Pipeline] Starting real-time mode...")
    #     print("[Pipeline] Press 'q' to quit, 's' to save frame, 'a' to toggle audio\n")
    #
    #     cap = cv2.VideoCapture(camera_id)
    #     if not cap.isOpened():
    #         print("[Pipeline] ERROR: Could not open camera")
    #         return
    #
    #     # Create results directory if needed
    #     os.makedirs("results", exist_ok=True)
    #
    #     frame_count = 0
    #     saved_frames = 0
    #
    #     try:
    #         while True:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 print("[Pipeline] No frame from camera")
    #                 break
    #
    #             frame_count += 1
    #
    #             # Resize for faster processing
    #             frame = cv2.resize(frame, (640, 480))
    #
    #             # Process frame
    #             result = self.process_frame(frame)
    #
    #             # Create visualization
    #             vis_frame = self.create_visualization(frame, result)
    #
    #             if self.enable_display:
    #                 cv2.imshow("Assistive Navigation System", vis_frame)
    #
    #             # Handle key presses
    #             key = cv2.waitKey(1) & 0xFF
    #             if key == ord('q'):
    #                 print("\n[Pipeline] Quitting...")
    #                 break
    #             elif key == ord('s'):
    #                 saved_frames += 1
    #                 output_path = f"results/frame_{frame_count}_{int(time.time())}.jpg"
    #                 cv2.imwrite(output_path, vis_frame)
    #                 print(f"[Pipeline] Saved: {output_path}")
    #             elif key == ord('a'):
    #                 self.enable_audio = not self.enable_audio
    #                 status = "ENABLED" if self.enable_audio else "DISABLED"
    #                 print(f"[Pipeline] Audio: {status}")
    #
    #             # Check max frames
    #             if max_frames and frame_count >= max_frames:
    #                 print(f"[Pipeline] Reached max frames ({max_frames})")
    #                 break
    #
    #     except KeyboardInterrupt:
    #         print("\n[Pipeline] Interrupted by user")
    #     finally:
    #         cap.release()
    #         cv2.destroyAllWindows()
    #         if self.audio:
    #             self.audio.shutdown()
    #
    #         # Print session statistics
    #         self.print_statistics()
    # ============================================================================

    def run_on_images(self, image_paths=None, image_folder="resources"):
        """
        Run the pipeline on images from a folder or list.

        Args:
            image_paths (list): List of image file paths (if None, uses image_folder)
            image_folder (str): Folder containing images (default: resources/)
        """
        print("\n[Pipeline] Starting image-based processing mode...")
        
        # Create results directory if needed
        os.makedirs("results", exist_ok=True)

        # Get image paths
        if image_paths is None:
            if not os.path.exists(image_folder):
                print(f"[Pipeline] ERROR: Folder not found: {image_folder}")
                return
            
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend([os.path.join(image_folder, f) 
                                   for f in os.listdir(image_folder) 
                                   if f.lower().endswith(ext.replace('*', ''))])
            image_paths = sorted(image_files)

        if not image_paths:
            print(f"[Pipeline] ERROR: No images found in {image_folder}")
            return

        print(f"[Pipeline] Found {len(image_paths)} images to process")

        frame_count = 0
        saved_frames = 0

        try:
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    print(f"[Pipeline] WARNING: Image not found: {image_path}")
                    continue

                frame_count += 1

                print(f"\n[Pipeline] Processing image {frame_count}/{len(image_paths)}: {os.path.basename(image_path)}")

                # Load image
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"[Pipeline] ERROR: Could not load image: {image_path}")
                    continue

                # Resize for consistent processing
                frame = cv2.resize(frame, (640, 480))

                # Process frame
                result = self.process_frame(frame)

                # Create visualization
                vis_frame = self.create_visualization(frame, result)

                if self.enable_display:
                    cv2.imshow("Assistive Navigation System - Image Mode", vis_frame)
                    print("[Pipeline] Press any key to continue to next image...")
                    cv2.waitKey(0)

                # Save result
                saved_frames += 1
                output_path = f"results/processed_{frame_count}_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
                cv2.imwrite(output_path, vis_frame)
                print(f"[Pipeline] Saved: {output_path}")

                # Print frame statistics
                if result['status'] == 'success':
                    print(f"  Objects detected: {len(result['detections'])}")
                    print(f"  Safety score: {result.get('safety_score', 0):.1f}/10")
                    print(f"  Guidance: {result.get('guidance', 'N/A')}")

        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted by user")
        finally:
            cv2.destroyAllWindows()
            if self.audio:
                self.audio.shutdown()

            # Print session statistics
            self.print_statistics()
            print(f"\n[Pipeline] Processed {frame_count} images, saved {saved_frames} results")

    def print_statistics(self):
        """Print session statistics."""
        elapsed = time.time() - self.start_time
        avg_frame_time = self.total_inference_time / self.frame_count if self.frame_count > 0 else 0

        print("\n" + "="*70)
        print("SESSION STATISTICS")
        print("="*70)
        print(f"Frames processed: {self.frame_count}")
        print(f"Total time: {elapsed:.1f} seconds")
        print(f"Average FPS: {self.get_fps():.2f}")
        print(f"Average frame time: {avg_frame_time*1000:.1f} ms")
        print("="*70 + "\n")

    def get_fps(self):
        """Calculate current FPS."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0


def main():
    """Main entry point for the assistive navigation system."""
    print("\n" + "="*70)
    print("ASSISTIVE NAVIGATION SYSTEM - IMAGE-BASED PROCESSING")
    print("Master's Thesis Project")
    print("="*70)

    # Initialize pipeline
    try:
        pipeline = AssistiveNavigationPipeline(
            enable_audio=True,
            enable_display=True,
            confidence_threshold=0.45
        )
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize pipeline: {e}")
        return

    # ===== IMAGE-BASED PROCESSING (NEW) =====
    # Run on images from resources folder
    pipeline.run_on_images(image_folder="resources")
    
    # ===== UNCOMMENT BELOW FOR CUSTOM IMAGE PATHS =====
    # pipeline.run_on_images(image_paths=[
    #     "resources/Aston Martin.jpg",
    #     # Add more images here
    # ])


if __name__ == "__main__":
    main()




