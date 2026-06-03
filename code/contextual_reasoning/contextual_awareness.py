"""
Contextual Awareness Engine (Reduced)
=====================================
This module USED to generate hard-coded if-else navigation messages.
That responsibility has been moved to:
    - `fusion_engine.py`  - turns YOLO+MiDaS+ORB into a structured context.
    - `prompt_engineering.py` - turns that context into a LLaVA prompt.
    - `llava_reasoning.py` - LLaVA produces the natural-language guidance.

What remains here is purely a passive spatial bookkeeper used by the
visualization overlay (status panel on the debug window). It produces
NO guidance text - the `navigation_guidance` field is intentionally
empty so any old caller relying on it will fall back silently and not
display stale rule-based strings.
"""

import numpy as np


class ContextualAwarenessEngine:
    """Spatial summary helper for visualization. NOT a reasoning engine."""

    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.left_boundary = frame_width * 0.33
        self.right_boundary = frame_width * 0.67
        self.top_boundary = frame_height * 0.33
        self.bottom_boundary = frame_height * 0.67

        print("[Context] Spatial summary helper initialized (reasoning moved to LLaVA)")

    # ----------------------- public bookkeeping ----------------------- #

    def analyze_detections(self, detections_with_depth, frame_shape=None):
        """Categorise detections by spatial zone & distance class.

        Returns:
            dict with `critical_objects`, `warning_objects`, `nearby_objects`,
            `far_objects`, `spatial_distribution`, `overall_safety_score`,
            and (always-empty) `navigation_guidance` + `alerts` fields kept
            for backward-compatible callers.
        """
        if frame_shape:
            self.frame_height, self.frame_width = frame_shape[:2]
            self.left_boundary = self.frame_width * 0.33
            self.right_boundary = self.frame_width * 0.67
            self.top_boundary = self.frame_height * 0.33
            self.bottom_boundary = self.frame_height * 0.67

        analysis = {
            "total_objects": len(detections_with_depth),
            "critical_objects": [],
            "warning_objects": [],
            "nearby_objects": [],
            "far_objects": [],
            "spatial_distribution": {"left": [], "center": [], "right": []},
            "alerts": [],                  # LLaVA owns this now - kept empty
            "navigation_guidance": "",     # LLaVA owns this now - kept empty
            # The legacy hand-tuned safety score has been REMOVED. It was a
            # rule-based subtraction from a fixed 10.0 floor that did not
            # reflect any genuine scene understanding. Downstream code
            # should not consume this field anymore; we leave it as None so
            # any unmigrated caller is forced to handle absence explicitly.
            "overall_safety_score": None,
        }

        if not detections_with_depth:
            return analysis

        for det in detections_with_depth:
            position = self._horizontal_zone(det.get("bbox", [0, 0, 0, 0]), det.get("center"))
            height = self._vertical_zone(det.get("bbox", [0, 0, 0, 0]))
            proximity = float(det.get("proximity", 0.0))
            proximity_label = str(det.get("proximity_label", "far"))

            entry = {
                "class": det.get("class", "unknown"),
                "confidence": det.get("confidence", 0.0),
                "proximity": proximity,
                "proximity_label": proximity_label,
                "position": position,
                "height": height,
            }

            # Bucket purely for the debug overlay - NO score arithmetic.
            if proximity_label == "near":
                analysis["critical_objects"].append(entry)
            elif proximity_label == "mid":
                analysis["warning_objects"].append(entry)
            else:
                analysis["far_objects"].append(entry)

            analysis["spatial_distribution"][position].append(entry)

        return analysis

    # ----------------------- helpers ----------------------- #

    def _horizontal_zone(self, bbox, center=None):
        if center:
            cx = center[0]
        else:
            x1, _, x2, _ = bbox
            cx = (x1 + x2) / 2
        if cx < self.left_boundary:
            return "left"
        if cx > self.right_boundary:
            return "right"
        return "center"

    def _vertical_zone(self, bbox):
        _, y1, _, y2 = bbox
        cy = (y1 + y2) / 2
        if cy < self.top_boundary:
            return "above"
        if cy > self.bottom_boundary:
            return "below"
        return "center"

    def get_obstacle_map(self, analysis, grid_size=3):
        """Simple 3x3 occupancy grid for the debug visualisation."""
        grid = np.zeros((grid_size, grid_size))
        for obj in analysis["critical_objects"] + analysis["warning_objects"]:
            col = {"left": 0, "center": 1, "right": 2}.get(obj["position"], 1)
            row = {"above": 0, "center": 1, "below": 2}.get(obj.get("height", "center"), 1)
            grid[row, col] += 1.0
        return grid


if __name__ == "__main__":
    eng = ContextualAwarenessEngine()
    print(eng.analyze_detections([], frame_shape=(480, 640, 3)))
