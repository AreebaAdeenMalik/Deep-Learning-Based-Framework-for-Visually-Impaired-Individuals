"""
YOLOv12 Object Detection Module
================================
Provides GPU-accelerated YOLOv12 inference for real-time object detection.
Outputs structured detection results with confidence scores and bounding boxes.

"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time


class YOLOv12Detector:
    """
    GPU-accelerated YOLOv12 object detector.
    Handles model loading, inference, and structured output generation.
    """

    def __init__(self, model_path="yolo12l.pt", confidence_threshold=0.45, device=None):
        """
        Initialize YOLOv12 detector.

        Args:
            model_path (str): Path to YOLOv12 weights (e.g., yolo12l.pt)
            confidence_threshold (float): Minimum confidence score for detections
            device (str): 'cuda', 'cpu', or None (auto-detect)
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"[YOLO] Initializing YOLOv12 on device: {self.device}")

        try:
            # Load YOLOv12 model
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"[YOLO] Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"[YOLO] ERROR loading model: {str(e)}")
            raise

        self.confidence_threshold = confidence_threshold
        self.model.eval()  # Set to evaluation mode

        # Performance tracking
        self.inference_times = []
        self.max_history = 30  # Keep last 30 inference times for FPS calculation

    def detect_objects(self, frame, return_annotated=False):
        """
        Run YOLOv12 inference on a single frame.

        Args:
            frame (np.ndarray): Input frame (BGR format from OpenCV)
            return_annotated (bool): If True, return annotated frame with bounding boxes

        Returns:
            list: List of detections with structured format:
                [
                    {
                        "class": "person",
                        "confidence": 0.92,
                        "bbox": [x1, y1, x2, y2],  # pixel coordinates
                        "center": (cx, cy),
                        "class_id": 0
                    },
                    ...
                ]
            np.ndarray: Annotated frame (if return_annotated=True)
        """
        start_time = time.time()

        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, device=self.device, verbose=False)

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > self.max_history:
                self.inference_times.pop(0)

            # Parse results
            detections = []
            annotated_frame = frame.copy() if return_annotated else None

            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get class names
                    class_names = result.names  # Dictionary of class_id: class_name

                    # Process each detection
                    for box in result.boxes:
                        # Extract box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = class_names.get(class_id, f"class_{class_id}")

                        # Calculate center
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        # Build detection dict
                        detection = {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "center": (int(cx), int(cy)),
                            "class_id": class_id
                        }
                        detections.append(detection)

                        # Draw on annotated frame if requested
                        if return_annotated:
                            # Draw bounding box
                            cv2.rectangle(
                                annotated_frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 255, 0),
                                2
                            )

                            # Draw label
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(
                                annotated_frame,
                                label,
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )

                            # Draw center point
                            cv2.circle(annotated_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        except Exception as e:
            print(f"[YOLO] Inference error: {str(e)}")
            detections = []

        if return_annotated:
            return detections, annotated_frame
        else:
            return detections

    def get_fps(self):
        """Calculate average FPS from recent inference times."""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def filter_detections(self, detections, classes_to_keep=None):
        """
        Filter detections by class name.

        Args:
            detections (list): List of detections from detect_objects()
            classes_to_keep (list): List of class names to keep (None = keep all)

        Returns:
            list: Filtered detections
        """
        if classes_to_keep is None:
            return detections

        return [d for d in detections if d["class"] in classes_to_keep]

    def get_object_statistics(self, detections):
        """
        Get statistics about detected objects.

        Args:
            detections (list): List of detections

        Returns:
            dict: Statistics including object counts and confidence ranges
        """
        if not detections:
            return {
                "total_objects": 0,
                "classes": {},
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0
            }

        class_counts = {}
        confidences = []

        for det in detections:
            class_name = det["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(det["confidence"])

        return {
            "total_objects": len(detections),
            "classes": class_counts,
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences)
        }


def run_yolo_inference(image_path):
    """Legacy function - runs YOLO on a single image."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    model = YOLO("yolo12l.pt")
    results = model(image_path, device=device)

    annotated_frame = results[0].plot()

    cv2.imshow("Original Image", cv2.imread(image_path))
    cv2.imshow("YOLO12 Detections", annotated_frame)

    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("results/result_yolo12.jpg", annotated_frame)
    print("Saved result to result_yolo12.jpg")


def run_yolo_detection_on_image(image_path, model_path="yolo12l.pt", confidence_threshold=0.45):
    """
    Run YOLOv12 detection on a single image file.

    Args:
        image_path (str): Path to input image
        model_path (str): Path to YOLOv12 weights
        confidence_threshold (float): Minimum confidence score
    """
    print(f"\n{'='*60}")
    print(f"YOLOv12 Detection on Single Image")
    print(f"{'='*60}")

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not load image from {image_path}")
        return

    print(f"Image shape: {frame.shape}")

    # Initialize detector
    detector = YOLOv12Detector(model_path=model_path, confidence_threshold=confidence_threshold)

    # Run detection
    detections, annotated_frame = detector.detect_objects(frame, return_annotated=True)

    # Print results
    print(f"\nDetected {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"  [{i+1}] {det['class']}: confidence={det['confidence']:.3f}, bbox={det['bbox']}")

    # Print statistics
    stats = detector.get_object_statistics(detections)
    print(f"\nStatistics:")
    print(f"  Total objects: {stats['total_objects']}")
    print(f"  Classes: {stats['classes']}")
    print(f"  Avg Confidence: {stats['avg_confidence']:.3f}")

    # Display
    cv2.imshow("YOLOv12 Detection", annotated_frame)
    print("\nPress any key to close display...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    output_path = "results/result_yolo12.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved result to {output_path}")


def run_yolo_detection_on_webcam(model_path="yolo12l.pt", confidence_threshold=0.45, display_fps=True):
    """
    DEPRECATED: Real-time YOLOv12 detection on webcam stream.
    System now uses image-based processing.
    
    This code is commented out but kept for reference.

    Args:
        model_path (str): Path to YOLOv12 weights
        confidence_threshold (float): Minimum confidence score
        display_fps (bool): Display FPS on frame
    """
    # print(f"\n{'='*60}")
    # print(f"YOLOv12 Real-time Detection (Webcam) - DEPRECATED")
    # print(f"{'='*60}")
    # print("Press 'q' to quit, 's' to save current frame")
    #
    # # Initialize detector
    # detector = YOLOv12Detector(model_path=model_path, confidence_threshold=confidence_threshold)
    #
    # # Open webcam
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("ERROR: Could not open webcam")
    #     return
    #
    # print("Webcam opened. Starting detection loop...")
    #
    # frame_count = 0
    # saved_frames = 0
    #
    # try:
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #
    #         frame_count += 1
    #
    #         # Run detection
    #         detections, annotated_frame = detector.detect_objects(frame, return_annotated=True)
    #
    #         # Draw FPS
    #         if display_fps:
    #             fps = detector.get_fps()
    #             cv2.putText(
    #                 annotated_frame,
    #                 f"FPS: {fps:.1f}",
    #                 (10, 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 1.0,
    #                 (0, 255, 0),
    #                 2
    #             )
    #
    #         # Draw object count
    #         cv2.putText(
    #             annotated_frame,
    #             f"Objects: {len(detections)}",
    #             (10, 70),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1.0,
    #             (0, 255, 0),
    #             2
    #         )
    #
    #         # Display
    #         cv2.imshow("YOLOv12 Real-time Detection", annotated_frame)
    #
    #         # Handle key presses
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             print("Quitting...")
    #             break
    #         elif key == ord('s'):
    #             saved_frames += 1
    #             output_path = f"results/yolo_frame_{frame_count}.jpg"
    #             cv2.imwrite(output_path, annotated_frame)
    #             print(f"Saved frame to {output_path}")
    #
    # except KeyboardInterrupt:
    #     print("\nInterrupted by user")
    # finally:
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     print(f"\nSession complete:")
    #     print(f"  Processed {frame_count} frames")
    #     print(f"  Saved {saved_frames} frames")
    #     print(f"  Average FPS: {detector.get_fps():.1f}")
    
    print("[YOLO] Webcam mode is deprecated - use image-based processing")
    print("[YOLO] Run main_pipeline.py for image processing")


if __name__ == "__main__":
    import global_config
    import os

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # ===== IMAGE-BASED PROCESSING (DEFAULT) =====
    # Test on single image
    image_path = global_config.image_path
    if os.path.exists(image_path):
        print(f"\nProcessing image: {image_path}")
        run_yolo_detection_on_image(image_path)
    else:
        print(f"Image not found at {image_path}.")
        print("Using resources/my_image.jpg.jpg instead...")
        alt_path = "resources/my_image.jpg.jpg"
        if os.path.exists(alt_path):
            run_yolo_detection_on_image(alt_path)
        else:
            print("No images found. Please add images to resources/ folder")
    
    # ===== WEBCAM MODE (DEPRECATED) =====
    # Uncomment below to use webcam mode (requires changes to enable_display logic)
    # run_yolo_detection_on_webcam()
