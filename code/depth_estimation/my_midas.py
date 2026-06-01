"""
MiDaS Monocular Depth Estimation Module
========================================
Provides GPU-accelerated real-time depth estimation from video frames.
Converts relative depth to approximate distance measures.

"""

import cv2
import torch
import os
import numpy as np
import time
import global_config


class MiDaSDepthEstimator:
    """
    Real-time monocular depth estimation using MiDaS v2.1.
    Provides depth maps and distance estimation at specific points.
    """

    def __init__(self, device=None):
        """
        Initialize MiDaS depth estimator.
        
        Args:
            device (str): 'cuda', 'cpu', or None (auto-detect)
        """
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"[MiDaS] Initializing on device: {self.device}")

        try:
            # Load model and transforms
            print("[MiDaS] Loading MiDaS v2.1 model...")
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.default_transform
            print("[MiDaS] Model loaded successfully")
        except Exception as e:
            print(f"[MiDaS] ERROR loading model: {str(e)}")
            raise

        # Performance tracking
        self.inference_times = []
        self.max_history = 30

    def estimate_depth(self, frame):
        """
        Estimate depth map for a single frame.
        
        Args:
            frame (np.ndarray): Input frame (BGR format from OpenCV)
            
        Returns:
            np.ndarray: Depth map (same resolution as input frame)
        """
        start_time = time.time()

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_batch = self.transform(frame_rgb).to(self.device)

            # Inference
            with torch.no_grad():
                prediction = self.model(input_batch)

                # Resize to original resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Convert to numpy
            depth_map = prediction.cpu().numpy()

            # Track inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > self.max_history:
                self.inference_times.pop(0)

            return depth_map

        except Exception as e:
            print(f"[MiDaS] Inference error: {str(e)}")
            return None

    def get_distance_at_point(self, depth_map, x, y, normalize=True):
        """
        Get estimated distance (in arbitrary units) at a specific pixel coordinate.
        
        Args:
            depth_map (np.ndarray): Depth map from estimate_depth()
            x (int): X pixel coordinate
            y (int): Y pixel coordinate
            normalize (bool): If True, normalize depth to 0-1 range
            
        Returns:
            float: Estimated distance/depth value at point
        """
        if depth_map is None or x < 0 or y < 0:
            return 0.0
        
        # Clip coordinates to valid range
        h, w = depth_map.shape
        x = min(max(int(x), 0), w - 1)
        y = min(max(int(y), 0), h - 1)
        
        depth_value = depth_map[y, x]
        
        # Normalize if requested: invert and scale to 0-10 meters
        if normalize:
            norm_depth = (1.0 / (1.0 + depth_value)) * 10.0
            return float(norm_depth)
        
        return float(depth_value)

    def get_distances_for_detections(self, depth_map, detections):
        """
        Get depth estimates for each detected object (using center point).
        
        Args:
            depth_map (np.ndarray): Depth map from estimate_depth()
            detections (list): Detection list with 'center' fields
            
        Returns:
            list: Updated detections with 'distance' field added
        """
        detections_with_depth = []
        
        for det in detections:
            det_copy = det.copy()
            if "center" in det:
                cx, cy = det["center"]
                distance = self.get_distance_at_point(depth_map, cx, cy, normalize=True)
                det_copy["distance"] = distance
            detections_with_depth.append(det_copy)
        
        return detections_with_depth

    def get_fps(self):
        """Calculate average FPS from recent inference times."""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def visualize_depth_map(self, depth_map, colormap=cv2.COLORMAP_MAGMA):
        """
        Create a color visualization of the depth map.
        
        Args:
            depth_map (np.ndarray): Depth map from estimate_depth()
            colormap (int): OpenCV colormap
            
        Returns:
            np.ndarray: RGB visualization
        """
        if depth_map is None:
            return None
        
        # Normalize depth map to 0-255
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min < 1e-6:
            depth_normalized = np.zeros_like(depth_map)
        else:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        
        depth_vis = (depth_normalized * 255).astype(np.uint8)
        depth_vis_color = cv2.applyColorMap(depth_vis, colormap)
        
        return depth_vis_color


def load_midas_model(device):
    """
    Legacy function for backward compatibility.
    Loads the MiDaS v2.1 model (CNN-based).
    """
    print("[MiDaS] Loading MiDaS v2.1 model (legacy)...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform

    return midas, transform


def run_depth_estimation(image_path):
    """Legacy function - runs depth estimation on single image."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    midas_model, midas_transform = load_midas_model(device)

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not open {image_path}")
        return

    img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    print("Running Depth Estimation...")
    input_batch = midas_transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas_model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    depth_map_vis = (depth_map_normalized * 255).astype(np.uint8)

    depth_map_vis = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_MAGMA)

    cv2.imshow("MiDaS Depth Map", depth_map_vis)

    print("\nPress any key to close the windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("results/midas_result.jpg", depth_map_vis)
    print("Saved result to midas_result.jpg")


if __name__ == "__main__":
    image_path = global_config.image_path

    if not os.path.exists(image_path):
        print(f"Creating dummy {image_path}...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy)

    run_depth_estimation(image_path)
