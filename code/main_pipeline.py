"""
Main Integrated Pipeline (LLaVA-centric)
========================================
Real-time assistive-navigation pipeline that uses LLaVA as the
contextual-reasoning brain. The exact data flow is:

    Camera Frame
        |
        v
    YOLOv12  ----- object detections (class, conf, bbox, center)
        |
        v
    MiDaS    ----- per-object distance (metres, approx.)
        |
        v
    ORB VO   ----- features, matches, score, ego-motion direction
        |
        v
    FusionEngine ----- FusedContext (objects + spatial zones + ego-motion
                                     + rule-based emergency_stop flag)
        |
        v
    PromptEngineering ----- structured text prompt
        |
        v
    LLaVA (image + prompt) ----- natural-language navigation instruction
        |
        v
    AudioGuidance (TTS) ----- spoken to the visually-impaired user

The ONLY rule-based shortcut is the latency-critical emergency stop,
which speaks `EMERGENCY_STOP_MESSAGE` directly without consulting LLaVA.

"""

import os
import time

import cv2
import numpy as np
import torch

import global_config
from Yolo12 import YOLOv12Detector
from my_midas import MiDaSDepthEstimator
from orb_slam_single_image import VisualOdometryORB, load_camera_params
from fusion_engine import FusionEngine
from prompt_engineering import build_prompt
from llava_reasoning import LlavaReasoningEngine
from contextual_awareness import ContextualAwarenessEngine  # used only for viz overlay
from audio_guidance import (
    AudioGuidanceSystem,
    EMERGENCY_STOP_MESSAGE,
    build_emergency_message,
)


class AssistiveNavigationPipeline:
    """End-to-end multimodal pipeline: YOLO + MiDaS + ORB + Fusion + LLaVA + TTS."""

    def __init__(
        self,
        enable_audio: bool = True,
        enable_display: bool = True,
        confidence_threshold: float = 0.45,
        llava_every_n_frames: int = 1,
    ):
        print("\n" + "=" * 70)
        print("ASSISTIVE NAVIGATION SYSTEM - INITIALIZATION (LLaVA contextual brain)")
        print("=" * 70)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[Pipeline] Device: {self.device}")
        if self.device == "cuda":
            print(f"[Pipeline] GPU: {torch.cuda.get_device_name(0)}")

        # -------------- perception modules -------------- #
        self.detector = YOLOv12Detector(
            model_path=getattr(global_config, "yolo_model_path", "yolo12l.pt"),
            confidence_threshold=confidence_threshold,
            device=self.device,
        )
        print("[Pipeline] YOLOv12 initialized")

        self.depth_estimator = MiDaSDepthEstimator(device=self.device)
        print("[Pipeline] MiDaS initialized")

        fx, fy, cx, cy = load_camera_params(
            getattr(global_config, "camera_config_path", "webcam_config.yaml")
        )
        self.vo = VisualOdometryORB(focal_len=fx, pp=(cx, cy))
        print("[Pipeline] ORB Visual Odometry initialized")

        # -------------- fusion + reasoning -------------- #
        self.fusion = FusionEngine(
            frame_width=getattr(global_config, "camera_width", 640),
            frame_height=getattr(global_config, "camera_height", 480),
            emergency_proximity_threshold=getattr(
                global_config, "emergency_proximity_threshold", 0.85),
            emergency_min_occupancy=getattr(
                global_config, "emergency_min_occupancy", 0.05),
            emergency_min_confidence=getattr(
                global_config, "emergency_min_confidence", 0.55),
        )
        print("[Pipeline] FusionEngine initialized")

        self.llava = LlavaReasoningEngine(
            model_id=getattr(global_config, "llava_model_id", None),
            device=self.device,
            load_in_4bit=getattr(global_config, "llava_load_in_4bit", True),
            max_new_tokens=getattr(global_config, "llava_max_new_tokens", 64),
            temperature=getattr(global_config, "llava_temperature", 0.2),
            do_sample=getattr(global_config, "llava_do_sample", False),
        )
        if self.llava.available:
            print("[Pipeline] LLaVA reasoning engine initialized")
        else:
            print("[Pipeline] WARNING: LLaVA unavailable - only emergency-stop guidance will be spoken.")

        self.awareness = ContextualAwarenessEngine(frame_width=640, frame_height=480)

        # -------------- audio -------------- #
        if enable_audio:
            self.audio = AudioGuidanceSystem(
                rate=getattr(global_config, "audio_rate", 160),
                volume=getattr(global_config, "audio_volume", 1.0),
                voice_id=getattr(global_config, "audio_voice_id", 0),
            )
            self.enable_audio = not self.audio.audio_disabled
        else:
            self.audio = None
            self.enable_audio = False

        self.enable_display = enable_display
        self.llava_every_n_frames = max(1, int(llava_every_n_frames))

        # Stats
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()

        # Cache last LLaVA guidance so we have something to overlay/announce
        # on frames where we deliberately SKIP the LLM call (real-time mode).
        self._last_guidance = ""
        self._last_guidance_frame = -1

        print("\n[Pipeline] Initialization complete.\n" + "=" * 70 + "\n")

    # ============================================================== #
    #                       per-frame processing                     #
    # ============================================================== #

    def process_frame(self, frame: np.ndarray) -> dict:
        if frame is None or not isinstance(frame, np.ndarray):
            return {
                "status": "error: invalid_frame",
                "frame_id": self.frame_count + 1,
                "detections": [], "depth_map": None, "localization": None,
                "fused_context": None, "guidance": "", "alerts": [],
            }

        frame_start = time.time()
        self.frame_count += 1

        result = {
            "frame_id": self.frame_count,
            "frame_shape": frame.shape,
            "detections": [],
            "depth_map": None,
            "localization": None,
            "fused_context": None,
            "prompt": "",
            "guidance": "",
            "guidance_source": "none",  # 'llava' | 'emergency' | 'cached' | 'none'
            "alerts": [],
            "timings": {},
            "status": "processing",
        }

        try:
            # --------- STEP 1: YOLOv12 --------- #
            t0 = time.time()
            detections = self.detector.detect_objects(frame, return_annotated=False)
            result["timings"]["yolo"] = time.time() - t0

            # --------- STEP 2: MiDaS --------- #
            t0 = time.time()
            depth_map = self.depth_estimator.estimate_depth(frame)
            result["timings"]["midas"] = time.time() - t0
            result["depth_map"] = depth_map

            # --------- STEP 3: fuse YOLO + MiDaS per object --------- #
            if depth_map is not None and detections:
                detections = self.depth_estimator.enrich_detections_with_proximity(
                    depth_map, detections
                )
            result["detections"] = detections

            # --------- STEP 4: ORB Visual Odometry --------- #
            t0 = time.time()
            localization = self.vo.process_frame(frame)
            result["timings"]["orb"] = time.time() - t0
            result["localization"] = localization

            # --------- STEP 5: Fusion Engine --------- #
            t0 = time.time()
            fused = self.fusion.fuse(
                frame_id=self.frame_count,
                frame_shape=frame.shape,
                detections=detections,
                localization=localization,
            )
            result["fused_context"] = fused
            result["timings"]["fusion"] = time.time() - t0

            # Spatial bookkeeping for the debug overlay (no scoring)
            result["context_analysis"] = self.awareness.analyze_detections(
                detections, frame_shape=frame.shape
            )
            # safety_score is intentionally None - the legacy rule-based
            # subtraction has been removed. We keep the key only so the
            # overlay code can detect its absence and skip rendering.
            result["safety_score"] = result["context_analysis"]["overall_safety_score"]

            # --------- STEP 6: latency-critical safety override --------- #
            # If multi-modal agreement raises the emergency flag, speak a
            # data-driven template IMMEDIATELY at top priority, but DO NOT
            # skip LLaVA - we still let it produce an enriched follow-up
            # so guidance is never purely templated.
            if fused.emergency_stop:
                nearest_obj = self._nearest_emergency_object(fused)
                emergency_msg = build_emergency_message(
                    label=nearest_obj.label if nearest_obj else None,
                    side=nearest_obj.horizontal_zone if nearest_obj else None,
                    proximity_label=(
                        nearest_obj.proximity_label if nearest_obj else None
                    ),
                )
                result["guidance"] = emergency_msg
                result["guidance_source"] = "emergency"
                result["alerts"] = [{
                    "level": "CRITICAL",
                    "message": fused.emergency_reason,
                    "priority": 10,
                }]
                if self.enable_audio and self.audio:
                    self.audio.emergency_speak(emergency_msg)

            # --------- STEP 7: Prompt engineering + LLaVA --------- #
            # LLaVA runs on EVERY eligible frame, including emergency frames.
            # On an emergency frame its output enriches (does not replace)
            # the spoken template, and the safety section in the prompt
            # tells LLaVA the override is active.
            if self.llava.available and self.frame_count % self.llava_every_n_frames == 0:
                prompt = build_prompt(fused)
                result["prompt"] = prompt

                t0 = time.time()
                guidance = self.llava.generate(frame, prompt)
                result["timings"]["llava"] = time.time() - t0

                if guidance:
                    # Don't overwrite the emergency utterance the user will
                    # hear first; surface LLaVA as an enrichment instead.
                    if result["guidance_source"] == "emergency":
                        result["llava_enrichment"] = guidance
                    else:
                        result["guidance"] = guidance
                        result["guidance_source"] = "llava"
                    self._last_guidance = guidance
                    self._last_guidance_frame = self.frame_count

            # If we skipped LLaVA this frame, reuse the most recent guidance
            if (
                not result["guidance"]
                and result["guidance_source"] != "emergency"
                and self._last_guidance
            ):
                result["guidance"] = self._last_guidance
                result["guidance_source"] = "cached"

            # --------- STEP 8: speak the LLaVA instruction --------- #
            # Only speak the LLaVA reply when no emergency template was
            # already spoken this frame (avoids two utterances stacking).
            if (
                self.enable_audio and self.audio
                and result["guidance_source"] == "llava"
                and result["guidance"]
            ):
                self.audio.speak(result["guidance"], priority=5)
            elif (
                self.enable_audio and self.audio
                and result["guidance_source"] == "emergency"
                and result.get("llava_enrichment")
            ):
                # Queue the enrichment behind the emergency utterance.
                self.audio.speak(result["llava_enrichment"], priority=6)

            result["status"] = "success"

        except Exception as e:
            print(f"[Pipeline] ERROR processing frame: {e}")
            result["status"] = f"error: {e}"

        result["total_time"] = time.time() - frame_start
        self.total_inference_time += result["total_time"]
        return result

    # ============================================================== #
    #                       helpers                                  #
    # ============================================================== #

    def _nearest_emergency_object(self, fused):
        """Return the FusedObject that triggered the emergency stop, or None.

        Re-applies the same multi-modal filter the fusion engine used so
        the spoken template references the actual culprit object instead
        of a generic obstacle. Picks the highest-proximity match.
        """
        if not fused or not fused.objects:
            return None
        candidates = [
            o for o in fused.objects
            if o.label in self.fusion.emergency_stop_classes
            and o.confidence >= self.fusion.emergency_min_confidence
            and o.horizontal_zone == "center"
            and o.occupancy_ratio >= self.fusion.emergency_min_occupancy
            and o.proximity >= self.fusion.emergency_proximity_threshold
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda o: o.proximity)

    # ============================================================== #
    #                       visualisation                            #
    # ============================================================== #

    def create_visualization(self, frame: np.ndarray, result: dict) -> np.ndarray:
        if frame is None or not isinstance(frame, np.ndarray):
            return np.zeros((480, 640, 3), dtype=np.uint8)
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Detections (colour and label use relative PROXIMITY, not metres)
        for det in result.get("detections", []):
            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            p = float(det.get("proximity", 0.0))
            p_label = det.get("proximity_label", "far")
            color = (
                (0, 0, 255) if p_label == "near"
                else (0, 165, 255) if p_label == "mid"
                else (0, 255, 0)
            )
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = (
                f"{det.get('class','?')}:{det.get('confidence',0):.2f} "
                f"{p_label}({p:.2f})"
            )
            cv2.putText(vis, label, (int(x1), max(15, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ORB keypoints
        loc = result.get("localization") or {}
        keypoints = loc.get("keypoints")
        if keypoints is not None and len(keypoints) > 0:
            for pt in keypoints:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)

        # Status panel (no rule-based 'safety score' anymore)
        cv2.putText(vis, f"Frame {result['frame_id']}  FPS {self.get_fps():.1f}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Objects: {len(result.get('detections', []))}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        fused_ctx = result.get("fused_context")
        if fused_ctx is not None and fused_ctx.nearest_object:
            cv2.putText(
                vis,
                f"Nearest: {fused_ctx.nearest_object} "
                f"({fused_ctx.nearest_object_proximity_label}, "
                f"p={fused_ctx.max_proximity:.2f}, {fused_ctx.nearest_object_position})",
                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )

        em = fused_ctx.ego_motion if fused_ctx is not None else None
        if em is not None:
            cv2.putText(
                vis,
                f"ORB: {em.status}  dir={em.direction}  loc_conf={em.localization_confidence:.2f}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1,
            )

        # Guidance (LLaVA) along the bottom
        guidance = result.get("guidance", "")
        src = result.get("guidance_source", "none")
        if guidance:
            color = (0, 0, 255) if src == "emergency" else (255, 255, 255)
            cv2.putText(vis, f"[{src.upper()}] {guidance[:90]}",
                        (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return vis

    # ============================================================== #
    #                       runners                                  #
    # ============================================================== #

    def run_on_images(self, image_paths=None, image_folder="resources"):
        print("\n[Pipeline] Image-based processing mode.")
        os.makedirs("results", exist_ok=True)

        if image_paths is None:
            if not os.path.exists(image_folder):
                print(f"[Pipeline] ERROR: folder not found: {image_folder}")
                return
            image_paths = sorted([
                os.path.join(image_folder, f) for f in os.listdir(image_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])

        if not image_paths:
            print(f"[Pipeline] No images found in {image_folder}")
            return
        print(f"[Pipeline] Found {len(image_paths)} images.")

        try:
            for i, path in enumerate(image_paths, 1):
                if not os.path.exists(path):
                    print(f"[Pipeline] missing: {path}")
                    continue
                print(f"\n[Pipeline] ({i}/{len(image_paths)}) {os.path.basename(path)}")
                frame = cv2.imread(path)
                if frame is None:
                    print(f"[Pipeline] cannot read: {path}")
                    continue
                frame = cv2.resize(frame, (640, 480))

                result = self.process_frame(frame)
                vis = self.create_visualization(frame, result)

                if self.enable_display:
                    cv2.imshow("Assistive Navigation System (LLaVA)", vis)
                    print("[Pipeline] Press any key for next image (q to quit).")
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        break

                out_path = f"results/processed_{i}_{os.path.splitext(os.path.basename(path))[0]}.jpg"
                cv2.imwrite(out_path, vis)
                print(f"[Pipeline] saved {out_path}")

                if result["status"] == "success":
                    print(f"  objects   : {len(result['detections'])}")
                    fc = result.get("fused_context")
                    if fc is not None and fc.nearest_object:
                        print(
                            f"  nearest   : {fc.nearest_object} "
                            f"({fc.nearest_object_proximity_label}, "
                            f"p={fc.max_proximity:.2f}, {fc.nearest_object_position})"
                        )
                    print(f"  source    : {result['guidance_source']}")
                    print(f"  guidance  : {result.get('guidance', '')}")
                    if result.get("llava_enrichment"):
                        print(f"  enrich    : {result['llava_enrichment']}")
                    t = result.get("timings", {})
                    print(
                        "  timings   : "
                        f"yolo={t.get('yolo',0)*1000:.0f}ms  "
                        f"midas={t.get('midas',0)*1000:.0f}ms  "
                        f"orb={t.get('orb',0)*1000:.0f}ms  "
                        f"fusion={t.get('fusion',0)*1000:.0f}ms  "
                        f"llava={t.get('llava',0)*1000:.0f}ms"
                    )
        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted.")
        finally:
            cv2.destroyAllWindows()
            if self.audio:
                self.audio.shutdown()
            self.print_statistics()

    def run_realtime(self, camera_id: int = 0, max_frames=None):
        """Real-time webcam mode.

        For a real-time experience LLaVA is called every Nth frame
        (`llava_every_n_frames`) so YOLO/MiDaS/ORB never block waiting
        for the VLM. Emergency stops still fire on every frame.
        """
        print("\n[Pipeline] Real-time mode (q to quit)")
        os.makedirs("results", exist_ok=True)
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("[Pipeline] ERROR: could not open camera")
            return

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (640, 480))
                result = self.process_frame(frame)
                vis = self.create_visualization(frame, result)
                if self.enable_display:
                    cv2.imshow("Assistive Navigation System (LLaVA)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if max_frames and self.frame_count >= max_frames:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.audio:
                self.audio.shutdown()
            self.print_statistics()

    # ============================================================== #
    #                       stats                                    #
    # ============================================================== #

    def get_fps(self) -> float:
        elapsed = time.time() - self.start_time
        return (self.frame_count / elapsed) if elapsed > 0 else 0.0

    def print_statistics(self):
        elapsed = time.time() - self.start_time
        avg = (self.total_inference_time / self.frame_count) if self.frame_count else 0
        print("\n" + "=" * 70)
        print("SESSION STATISTICS")
        print("=" * 70)
        print(f"Frames processed : {self.frame_count}")
        print(f"Total wall time  : {elapsed:.1f}s")
        print(f"Average FPS      : {self.get_fps():.2f}")
        print(f"Average / frame  : {avg*1000:.1f} ms")
        if hasattr(self, "llava") and self.llava.available:
            print(f"LLaVA avg latency: {self.llava.average_latency()*1000:.1f} ms  "
                  f"calls={self.llava.call_count}")
        print("=" * 70 + "\n")


def main():
    print("\n" + "=" * 70)
    print("ASSISTIVE NAVIGATION SYSTEM (LLaVA-centric)")
    print("Master's Thesis Project")
    print("=" * 70)

    try:
        pipeline = AssistiveNavigationPipeline(
            enable_audio=getattr(global_config, "enable_audio", True),
            enable_display=getattr(global_config, "enable_display", True),
            confidence_threshold=getattr(global_config, "yolo_confidence_threshold", 0.45),
            llava_every_n_frames=getattr(global_config, "llava_every_n_frames", 1),
        )
    except Exception as e:
        print(f"FATAL: pipeline init failed: {e}")
        return

    pipeline.run_on_images(image_folder=getattr(global_config, "image_folder", "resources"))


if __name__ == "__main__":
    main()
