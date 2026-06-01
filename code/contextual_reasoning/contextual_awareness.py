"""
Contextual Awareness Engine
============================
Analyzes detected objects and spatial relationships to provide intelligent safety alerts.
Determines obstacle direction, proximity warnings, and navigation guidance.
"""

import numpy as np


class ContextualAwarenessEngine:
    """
    Analyzes multi-modal input (object detections + depth) to generate contextual alerts.
    Determines object positions (left/center/right, near/mid/far) and safety priorities.
    """

    def __init__(self, frame_width=640, frame_height=480):
        """
        Initialize contextual awareness engine.

        Args:
            frame_width (int): Camera frame width in pixels
            frame_height (int): Camera frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define spatial zones (horizontal)
        self.left_boundary = frame_width * 0.33
        self.right_boundary = frame_width * 0.67
        
        # Define spatial zones (vertical)
        self.top_boundary = frame_height * 0.33
        self.bottom_boundary = frame_height * 0.67
        
        # Distance thresholds (in meters)
        self.critical_distance = 0.5      # Very close, immediate warning
        self.warning_distance = 1.5       # Close, prepare warning
        self.attention_distance = 3.0     # Moderate distance, information
        
        # Dangerous/priority objects
        self.critical_classes = {'person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck'}
        self.warning_classes = {'stairs', 'door', 'chair', 'table', 'wall', 'pole'}
        
        # Recent detections for temporal coherence
        self.last_alerts = {}
        self.alert_cooldown = 1.0  # Seconds between same alert type
        
        print("[Context] Awareness engine initialized")

    def determine_position(self, bbox, center=None):
        """
        Determine horizontal position of object (left/center/right).

        Args:
            bbox (list): Bounding box [x1, y1, x2, y2]
            center (tuple): Center point (cx, cy) - if provided, used instead of bbox

        Returns:
            str: Position ('left', 'center', or 'right')
        """
        if center:
            cx = center[0]
        else:
            # Calculate from bbox
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2

        if cx < self.left_boundary:
            return "left"
        elif cx > self.right_boundary:
            return "right"
        else:
            return "center"

    def determine_height(self, bbox):
        """
        Determine vertical position of object (above/center/below eye level).

        Args:
            bbox (list): Bounding box [x1, y1, x2, y2]

        Returns:
            str: Position ('above', 'center', or 'below')
        """
        y1, y2 = bbox[1], bbox[3]
        cy = (y1 + y2) / 2

        if cy < self.top_boundary:
            return "above"
        elif cy > self.bottom_boundary:
            return "below"
        else:
            return "center"

    def classify_distance(self, distance_m):
        """
        Classify distance into categories for alert generation.

        Args:
            distance_m (float): Distance in meters

        Returns:
            str: ('critical', 'warning', 'attention', or 'far')
        """
        if distance_m < self.critical_distance:
            return "critical"
        elif distance_m < self.warning_distance:
            return "warning"
        elif distance_m < self.attention_distance:
            return "attention"
        else:
            return "far"

    def get_safety_priority(self, detection):
        """
        Calculate safety priority of a detection (0-10 scale).

        Args:
            detection (dict): Detection with 'class', 'confidence', 'distance'

        Returns:
            float: Priority score (0-10, higher = more urgent)
        """
        priority = 0.0
        
        obj_class = detection.get('class', '').lower()
        distance = detection.get('distance', 999.0)
        confidence = detection.get('confidence', 0.0)
        
        # Base priority based on object class
        if obj_class in self.critical_classes:
            priority += 6.0  # High priority for moving objects
        elif obj_class in self.warning_classes:
            priority += 4.0  # Medium priority for static obstacles
        else:
            priority += 2.0  # Low priority for neutral objects
        
        # Boost priority based on distance (closer = more urgent)
        if distance < self.critical_distance:
            priority += 4.0
        elif distance < self.warning_distance:
            priority += 2.0
        elif distance < self.attention_distance:
            priority += 0.5
        
        # Boost based on confidence
        priority += confidence
        
        return min(priority, 10.0)

    def analyze_detections(self, detections_with_depth, frame_shape=None):
        """
        Analyze all detections and generate contextual information.

        Args:
            detections_with_depth (list): Detections with 'distance' field
            frame_shape (tuple): Frame shape (height, width) for updating zones

        Returns:
            dict: Comprehensive analysis including alerts and spatial info
        """
        if frame_shape:
            self.frame_height, self.frame_width = frame_shape[:2]

        analysis = {
            "total_objects": len(detections_with_depth),
            "critical_objects": [],
            "warning_objects": [],
            "nearby_objects": [],
            "far_objects": [],
            "spatial_distribution": {
                "left": [],
                "center": [],
                "right": []
            },
            "alerts": [],
            "navigation_guidance": "",
            "overall_safety_score": 10.0
        }

        if not detections_with_depth:
            analysis["navigation_guidance"] = "Path is clear"
            return analysis

        # Analyze each detection
        for det in detections_with_depth:
            priority = self.get_safety_priority(det)
            position = self.determine_position(det.get('bbox', [0, 0, 0, 0]), det.get('center'))
            distance_class = self.classify_distance(det.get('distance', 999.0))
            
            # Build analysis entry
            analysis_entry = {
                "class": det.get('class', 'unknown'),
                "confidence": det.get('confidence', 0.0),
                "distance": det.get('distance', 999.0),
                "position": position,
                "height": self.determine_height(det.get('bbox', [0, 0, 0, 0])),
                "distance_class": distance_class,
                "priority": priority
            }
            
            # Categorize by distance
            if distance_class == "critical":
                analysis["critical_objects"].append(analysis_entry)
                analysis["overall_safety_score"] -= 3.0
            elif distance_class == "warning":
                analysis["warning_objects"].append(analysis_entry)
                analysis["overall_safety_score"] -= 1.5
            elif distance_class == "attention":
                analysis["nearby_objects"].append(analysis_entry)
                analysis["overall_safety_score"] -= 0.5
            else:
                analysis["far_objects"].append(analysis_entry)
            
            # Spatial distribution
            analysis["spatial_distribution"][position].append(analysis_entry)

        # Generate alerts based on analysis
        alerts = self._generate_alerts(analysis)
        analysis["alerts"] = alerts
        
        # Generate navigation guidance
        guidance = self._generate_guidance(analysis)
        analysis["navigation_guidance"] = guidance
        
        # Clamp safety score
        analysis["overall_safety_score"] = max(0.0, min(10.0, analysis["overall_safety_score"]))
        
        return analysis

    def _generate_alerts(self, analysis):
        """
        Generate prioritized alerts based on analysis.

        Args:
            analysis (dict): Output from analyze_detections()

        Returns:
            list: List of alerts sorted by priority
        """
        alerts = []
        
        # Critical alerts
        for obj in analysis["critical_objects"]:
            alert = {
                "level": "CRITICAL",
                "message": f"IMMEDIATE: {obj['class']} {obj['distance']:.1f}m {obj['position']}",
                "priority": 10,
                "object": obj['class'],
                "distance": obj['distance'],
                "position": obj['position']
            }
            alerts.append(alert)
        
        # Warning alerts
        for obj in analysis["warning_objects"]:
            alert = {
                "level": "WARNING",
                "message": f"{obj['class']} approaching, {obj['distance']:.1f}m {obj['position']}",
                "priority": 7,
                "object": obj['class'],
                "distance": obj['distance'],
                "position": obj['position']
            }
            alerts.append(alert)
        
        # Attention alerts (only if high confidence and priority)
        for obj in analysis["nearby_objects"][:3]:  # Limit to top 3
            if obj.get('priority', 0) >= 4.0:
                alert = {
                    "level": "ATTENTION",
                    "message": f"{obj['class']} {obj['distance']:.1f}m",
                    "priority": 5,
                    "object": obj['class'],
                    "distance": obj['distance']
                }
                alerts.append(alert)
        
        # Sort by priority
        alerts.sort(key=lambda x: x['priority'], reverse=True)
        
        return alerts

    def _generate_guidance(self, analysis):
        """
        Generate navigation guidance based on spatial analysis.

        Args:
            analysis (dict): Output from analyze_detections()

        Returns:
            str: Navigation guidance message
        """
        critical_count = len(analysis["critical_objects"])
        warning_count = len(analysis["warning_objects"])
        
        # Critical obstruction
        if critical_count > 0:
            return "STOP - Critical obstacle ahead"
        
        # Multiple warnings
        if warning_count >= 3:
            return "Caution - Multiple obstacles detected"
        
        # Check directions for obstacles
        left_objects = analysis["spatial_distribution"]["left"]
        right_objects = analysis["spatial_distribution"]["right"]
        center_objects = analysis["spatial_distribution"]["center"]
        
        # Suggest movement
        if len(center_objects) > len(left_objects) and len(center_objects) > len(right_objects):
            return "Move left or right to avoid center obstacles"
        elif len(left_objects) > len(right_objects):
            return "Move to the right for clearer path"
        elif len(right_objects) > len(left_objects):
            return "Move to the left for clearer path"
        
        # Default messages
        if warning_count > 0:
            return "Proceed with caution"
        elif len(analysis["nearby_objects"]) > 0:
            return "Path mostly clear, objects detected"
        else:
            return "Path is clear"

    def get_obstacle_map(self, analysis, grid_size=3):
        """
        Create a simple grid-based obstacle map for spatial understanding.

        Args:
            analysis (dict): Output from analyze_detections()
            grid_size (int): Size of grid (default 3x3)

        Returns:
            np.ndarray: Grid showing obstacle densities
        """
        grid = np.zeros((grid_size, grid_size))
        
        for obj in analysis["critical_objects"] + analysis["warning_objects"]:
            position = obj['position']
            height = obj.get('height', 'center')
            
            # Map position to grid coordinates
            col = 0 if position == 'left' else (2 if position == 'right' else 1)
            row = 0 if height == 'above' else (2 if height == 'below' else 1)
            
            grid[row, col] += 1.0
        
        return grid


def demo_contextual_awareness():
    """
    Demonstration of contextual awareness engine.
    """
    print("\n" + "="*60)
    print("Contextual Awareness Engine Demo")
    print("="*60)
    
    # Initialize engine
    engine = ContextualAwarenessEngine(frame_width=640, frame_height=480)
    
    # Create mock detections
    mock_detections = [
        {
            "class": "person",
            "confidence": 0.95,
            "bbox": [100, 150, 200, 300],
            "center": (150, 225),
            "distance": 1.2
        },
        {
            "class": "chair",
            "confidence": 0.87,
            "bbox": [400, 200, 500, 350],
            "center": (450, 275),
            "distance": 2.5
        },
        {
            "class": "car",
            "confidence": 0.92,
            "bbox": [300, 100, 550, 250],
            "center": (425, 175),
            "distance": 3.2
        }
    ]
    
    # Analyze
    analysis = engine.analyze_detections(mock_detections, frame_shape=(480, 640, 3))
    
    # Print results
    print(f"\nTotal Objects: {analysis['total_objects']}")
    print(f"Safety Score: {analysis['overall_safety_score']:.1f}/10")
    print(f"\nCritical Objects: {len(analysis['critical_objects'])}")
    print(f"Warning Objects: {len(analysis['warning_objects'])}")
    print(f"Nearby Objects: {len(analysis['nearby_objects'])}")
    
    print(f"\nAlerts:")
    for alert in analysis['alerts']:
        print(f"  [{alert['level']}] {alert['message']}")
    
    print(f"\nGuidance: {analysis['navigation_guidance']}")
    
    print("\nSpatial Distribution:")
    for direction, objs in analysis['spatial_distribution'].items():
        print(f"  {direction.capitalize()}: {len(objs)} objects")
        for obj in objs:
            print(f"    - {obj['class']} @ {obj['distance']:.1f}m")


if __name__ == "__main__":
    demo_contextual_awareness()
