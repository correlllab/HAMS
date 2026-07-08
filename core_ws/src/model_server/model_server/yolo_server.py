#!/usr/bin/env python3
"""
yolo_server — ROS 2 pub/sub node for open-vocabulary object detection.

Subscribes to one or more `sensor_msgs/CompressedImage` topics and, on a fixed
timer (default 1 Hz), runs the most recent frame of each topic through a
self-contained Ultralytics YOLO-World wrapper (see YoloWorld below) and publishes
a `custom_ros_messages/DetectionBundle` on that subscriber's OWN output topic —
one detection topic per image topic. Each bundle echoes the source image in
`rgb_image` and carries every detection as a `Detection` (class string, xyxy box
as bbox_min/bbox_max points, confidence).

Alongside each detection topic it also publishes an annotated `sensor_msgs/Image`
(the frame with the detection boxes drawn on it) on a sibling topic ending in
`/annotated` — add that as an Image display in RViz to see the detections.

The class vocabulary is open-vocabulary and set from the `queries` parameter
(model.set_classes(queries)), read live each cycle so `ros2 param set` retunes the
classes without a restart. All subscribers share the one vocabulary.

Parameters:
  image_topics      string[]  input CompressedImage topics to subscribe to.
  detection_topics  string[]  output DetectionBundle topics, parallel to
                              image_topics. If empty (or a length mismatch), each
                              output is derived as "<image_topic>/detections".
  queries           string[]  open-vocabulary class prompts (e.g. ['bottle','cup']).
  publish_rate_hz   double    per-topic inference/publish rate (default 1.0).
  conf_threshold    double    detection confidence threshold (default 0.90).
  nms_iou           double    NMS IoU threshold (default 0.01).
  enable_logging / enable_visualization / clear_logs — see model_logging.
"""
import os
import threading
from functools import partial

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point

from ultralytics import YOLOWorld

from custom_ros_messages.msg import Detection, DetectionBundle

from model_server.model_logging import ModelLogger, declare_logging_params

# ----- config -----
# YOLO-World weights live under <package>/weights/. `yolo_world_battery_best.pt`
# is our fine-tuned open-vocabulary checkpoint for the battery workcell (a full
# Ultralytics export, loaded directly). If the file is absent, ultralytics instead
# fetches the base checkpoint by this name on first use (downloads to the CWD); a
# checkpoint dict carrying a 'state_dict' is treated as a fine-tuned export and
# loaded on top of the base architecture.
YOLO_MODEL = "yolo_world_battery_best.pt"
# Confidence threshold: the model drops detections below this. High by default
# (matches the reference config) so only strong detections come back — lower it
# for higher recall.
YOLO_CONF_THRESHOLD = 0.90
# NMS IoU threshold: overlapping boxes above this IoU are merged. Low (0.01) so
# near-duplicate boxes for the same object collapse aggressively.
YOLO_NMS_IOU = 0.01
# Run inference in FP32. YOLO-World's set_classes() stores the CLIP text features
# (txt_feats) as FP32; with half=True the image branch runs FP16 and the detection
# head then multiplies a Half activation by a Float text embedding, raising
# "expected scalar type Float but found Half". Keeping the whole graph FP32 avoids
# that dtype mismatch. At ~1 Hz across a few cameras the FP16 speedup is irrelevant.
YOLO_HALF = False

# Defaults for the pub/sub wiring (all overridable via ROS params, see below).
# NB: the head camera publishes color on .../color/image_raw/compressed, but the
# hand cameras only publish RECTIFIED color (.../color/image_rect_raw/compressed) —
# their image_raw/compressed topic has no publisher. Match each accordingly.
DEFAULT_IMAGE_TOPICS = [
    "/realsense/head/color/image_raw/compressed",
    "/realsense/left_hand/color/image_rect_raw/compressed",
    "/realsense/right_hand/color/image_rect_raw/compressed",
]
# Default open-vocabulary class prompts: the battery-workcell parts the
# yolo_world_battery_best.pt checkpoint was fine-tuned on. Override live via the
# `queries` param (ros2 param set /yolo_server queries "[...]"). Names must match
# the checkpoint's training labels exactly.
DEFAULT_QUERIES = ["Bolt", "BusBar", "InteriorScrew", "Nut", "OrageCover",
                   "Screw", "ScrewHole"]
DEFAULT_PUBLISH_RATE_HZ = 1.0
# Suffix appended to an image topic to derive its detection topic when
# `detection_topics` is not supplied explicitly.
DETECTION_TOPIC_SUFFIX = "/detections"
# Suffix for the per-camera annotated-image topic (raw sensor_msgs/Image with the
# detection boxes drawn on it) — add this as an Image display in RViz.
ANNOTATED_TOPIC_SUFFIX = "/annotated"

# Weights resolve relative to this package (works with `colcon --symlink-install`,
# which symlinks the installed module back to the source tree). realpath (not
# abspath) is required to follow that symlink. Override the root with
# MODEL_SERVER_WEIGHTS_DIR.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_WEIGHTS_DIR = os.environ.get("MODEL_SERVER_WEIGHTS_DIR",
                              os.path.join(_PKG_ROOT, "weights"))


class YoloWorld:
    """Minimal Ultralytics YOLO-World wrapper for the yolo_detect service.

    Loads the model once (from <weights>/YOLO_MODEL if present, else by name so
    ultralytics downloads it), moves it to CUDA when available (FP16 inference is
    requested per call in `detect`), and detects a per-call, text-prompted class
    vocabulary via `detect`.
    """

    def __init__(self):
        weight_path = os.path.join(_WEIGHTS_DIR, YOLO_MODEL)
        if os.path.exists(weight_path):
            # A raw ultralytics checkpoint loads directly; a {'state_dict': ...}
            # export is a fine-tune, loaded on top of the base arch (non-strict so
            # the text-encoder / head buffers that aren't in the export are kept).
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.model = YOLOWorld(YOLO_MODEL)
                self.model.model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                self.model = YOLOWorld(weight_path)
        else:
            self.model = YOLOWorld(YOLO_MODEL)  # ultralytics downloads by name

        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))
        # Move to the target device eagerly so GPU/OOM errors surface at startup.
        # Inference runs FP32 (see YOLO_HALF): YOLO-World's FP32 text features are
        # incompatible with an FP16 image branch, so we do NOT half() the module and
        # pass half=False to predict() in detect().
        self.model.to(self.device)
        print(f"[YoloWorld init] Initialized on device {self.device}")

    def detect(self, bgr_img, queries, conf=YOLO_CONF_THRESHOLD, iou=YOLO_NMS_IOU,
               debug=False):
        """Open-vocabulary detection on one image.

        `bgr_img` is an HxWx3 uint8 BGR array (Ultralytics' numpy-input convention).
        `queries` is the list of text class prompts. Returns (boxes, scores,
        class_ids) as numpy arrays ranked best-first by score:
          - boxes:     (N, 4) float32, pixel xyxy
          - scores:    (N,) float32, in [0, 1]
          - class_ids: (N,) int32, index into `queries`
        Returns three empty arrays when nothing is detected.
        """
        self.model.set_classes(queries)
        with torch.no_grad():
            # FP32 (half=YOLO_HALF, False): set_classes stores FP32 text features,
            # so an FP16 image branch would raise "expected Float but found Half".
            results = self.model.predict(bgr_img, show=False, verbose=debug,
                                         conf=conf, iou=iou, half=YOLO_HALF)[0]
        boxes = results.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = results.boxes.conf.detach().cpu().numpy().astype(np.float32)
        class_ids = results.boxes.cls.detach().cpu().numpy().astype(np.int32)
        # Rank best-first so callers can take the top detections directly.
        order = np.argsort(-scores)
        return boxes[order], scores[order], class_ids[order]


# Max time a cycle waits for the YOLO model lock before giving up. A normal
# inference is well under a second; if the lock is held far longer a prior call
# has wedged (CUDA stall/OOM). Failing fast keeps the node responsive with a clear
# error instead of silently queueing every future cycle on a dead lock.
LOCK_ACQUIRE_TIMEOUT_SEC = 20.0


class _Channel:
    """One image-topic → detection-topic pipe: caches the latest frame and owns
    the publisher + timer that run inference on it."""

    def __init__(self, image_topic, detection_topic, annotated_topic,
                 publisher, annotated_publisher):
        self.image_topic = image_topic
        self.detection_topic = detection_topic
        self.annotated_topic = annotated_topic
        self.publisher = publisher                    # DetectionBundle
        self.annotated_publisher = annotated_publisher  # sensor_msgs/Image
        self._img_lock = threading.Lock()
        self._latest = None  # most recent CompressedImage, or None until one arrives

    def set_latest(self, msg):
        with self._img_lock:
            self._latest = msg

    def take_latest(self):
        with self._img_lock:
            return self._latest


class YoloServer(Node):
    def __init__(self):
        super().__init__("yolo_server")
        self.bridge = CvBridge()
        log, viz, clear = declare_logging_params(self)
        self.logger = ModelLogger(self, 'yolo', 'model_server', __file__,
                                  log=log, visualize=viz, clear=clear)

        # ----- parameters -----
        str_array = ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        image_topics = list(self.declare_parameter(
            "image_topics", DEFAULT_IMAGE_TOPICS, str_array).value)
        detection_topics = list(self.declare_parameter(
            "detection_topics", [], str_array).value)
        # queries is read LIVE each cycle (below) so it can be retuned at runtime;
        # defaults to the battery-workcell classes the checkpoint was trained on.
        self.declare_parameter("queries", DEFAULT_QUERIES, str_array)
        self.conf = float(self.declare_parameter(
            "conf_threshold", YOLO_CONF_THRESHOLD).value)
        self.iou = float(self.declare_parameter("nms_iou", YOLO_NMS_IOU).value)
        rate = float(self.declare_parameter(
            "publish_rate_hz", DEFAULT_PUBLISH_RATE_HZ).value)

        if not image_topics:
            raise RuntimeError("yolo_server: 'image_topics' is empty — "
                               "nothing to subscribe to")
        if rate <= 0.0:
            raise RuntimeError(f"yolo_server: 'publish_rate_hz' must be > 0 "
                               f"(got {rate})")
        # Pair each image topic with its output topic. An explicit, length-matching
        # `detection_topics` wins; otherwise every output is derived by suffix.
        if len(detection_topics) != len(image_topics):
            if detection_topics:
                self.get_logger().warn(
                    f"'detection_topics' ({len(detection_topics)}) does not match "
                    f"'image_topics' ({len(image_topics)}); deriving all outputs "
                    f"as <image_topic>{DETECTION_TOPIC_SUFFIX}")
            detection_topics = [self._derive_detection_topic(t)
                                for t in image_topics]

        # set_classes mutates model state that predict reads, so concurrent cycles
        # (one timer per topic, all firing independently) must serialize.
        self._lock = threading.RLock()
        self.get_logger().info("yolo_server loading YOLO-World ...")
        self.yolo = YoloWorld()  # eager load: surfaces weight/GPU errors at startup

        # One subscription + publisher + timer per image topic. A reentrant group
        # lets the timers fire concurrently under the MultiThreadedExecutor; the
        # model lock keeps the actual inference serialized.
        self._cb_group = ReentrantCallbackGroup()
        self._channels = []
        period = 1.0 / rate
        for img_topic, det_topic in zip(image_topics, detection_topics):
            ann_topic = self._derive_annotated_topic(img_topic)
            pub = self.create_publisher(DetectionBundle, det_topic, 10)
            ann_pub = self.create_publisher(Image, ann_topic, 10)
            ch = _Channel(img_topic, det_topic, ann_topic, pub, ann_pub)
            self._channels.append(ch)
            self.create_subscription(
                CompressedImage, img_topic, partial(self._on_image, ch),
                qos_profile_sensor_data, callback_group=self._cb_group)
            self.create_timer(period, partial(self._on_timer, ch),
                              callback_group=self._cb_group)
            self.get_logger().info(
                f"  {img_topic}  ->  {det_topic}  (+ {ann_topic})")

        self.get_logger().info(
            f"yolo_server ready: {len(self._channels)} topic(s) @ {rate:.2f} Hz")

    @staticmethod
    def _derive_detection_topic(image_topic):
        """Default output topic for an image topic: strip a trailing slash and
        append the detection suffix (e.g. '.../compressed' -> '.../compressed/detections')."""
        return image_topic.rstrip("/") + DETECTION_TOPIC_SUFFIX

    @staticmethod
    def _derive_annotated_topic(image_topic):
        """Annotated-image topic for an image topic: a sibling of the detection
        topic ending in '/annotated' (e.g. '.../compressed/annotated')."""
        return image_topic.rstrip("/") + ANNOTATED_TOPIC_SUFFIX

    def _current_queries(self):
        """The class vocabulary, read live so `ros2 param set` retunes it."""
        return [str(q) for q in self.get_parameter("queries").value]

    def _on_image(self, channel, msg):
        """Cache the latest frame for `channel`; inference happens on the timer."""
        channel.set_latest(msg)

    def _on_timer(self, channel):
        """One inference/publish cycle for a single image topic."""
        msg = channel.take_latest()
        if msg is None:
            # No frame yet on this topic — surface it (throttled) so a silent
            # camera is diagnosable instead of the channel just doing nothing.
            self.get_logger().warn(
                f"{channel.image_topic}: no frames received yet — is the camera "
                "publishing on this topic?", throttle_duration_sec=5.0)
            return

        queries = self._current_queries()
        if not queries:
            self.get_logger().warn(
                "no 'queries' set — skipping detection (set the queries param)",
                throttle_duration_sec=10.0)
            return

        rec = self.logger.start()
        rec.set(topic=channel.image_topic, queries=queries, n_queries=len(queries))
        try:
            # Ultralytics treats a numpy source as BGR (OpenCV order), so decode BGR.
            bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(
                f"{channel.image_topic}: failed to decode image: {e}")
            rec.finish(success=False, message=f"decode error: {e}")
            return
        rec.save_image('input', bgr)

        if not self._lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT_SEC):
            self.get_logger().warn(
                f"YOLO busy: model lock held > {LOCK_ACQUIRE_TIMEOUT_SEC:.0f}s "
                "(prior inference wedged or GPU overloaded); skipping cycle")
            rec.finish(success=False, message="model lock timeout")
            return
        try:
            boxes, scores, class_ids = self.yolo.detect(
                bgr, queries, conf=self.conf, iou=self.iou)
        except Exception as e:
            self.get_logger().error(f"YOLO detect failed: {e}")
            rec.finish(success=False, message=f"detection error: {e}")
            return
        finally:
            self._lock.release()

        bundle = self._build_bundle(msg, boxes, scores, class_ids, queries)
        channel.publisher.publish(bundle)

        n = int(boxes.shape[0])
        # Metrics: total detections + a per-query hit count (best_score too).
        per_query = {}
        for cid in class_ids.tolist():
            if 0 <= cid < len(queries):
                per_query[queries[cid]] = per_query.get(queries[cid], 0) + 1
        H, W = bgr.shape[:2]
        rec.set(n_detections=n, detections_per_query=per_query,
                best_score=float(scores[0]) if n else 0.0,
                image_hw=[int(H), int(W)])
        self.get_logger().info(
            f"{channel.detection_topic}: {n} detection(s) for "
            f"{len(queries)} quer(ies) per_query={per_query}" if n else
            f"{channel.detection_topic}: no detections for queries={queries}")

        # Draw the boxes once, publish the annotated frame for RViz, and (if the
        # visualize toggle is on) also save it to the model log dir.
        vis = self._draw_overlay(bgr, boxes, scores, class_ids, queries)
        if vis is not None:
            self._publish_annotated(channel, vis, msg.header)
            if self.logger.visualize:
                self._save_overlay_to_disk(rec, vis)
        rec.finish(success=True, message="ok")

    def _build_bundle(self, image_msg, boxes, scores, class_ids, queries):
        """Assemble a DetectionBundle: echo the source image and turn each detection
        into a Detection (class string + xyxy box as bbox_min/bbox_max points)."""
        bundle = DetectionBundle()
        # Echo the source frame so subscribers can correlate detections to pixels;
        # its header carries the original stamp + frame_id. depth_image, camera_info
        # and camera_pose are left at defaults (this node only has the RGB frame).
        bundle.rgb_image = image_msg
        detections = []
        for i in range(int(boxes.shape[0])):
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            cid = int(class_ids[i])
            det = Detection()
            det.cls = queries[cid] if 0 <= cid < len(queries) else str(cid)
            det.bbox_min = Point(x=x1, y=y1, z=0.0)
            det.bbox_max = Point(x=x2, y=y2, z=0.0)
            det.prob = float(scores[i])
            detections.append(det)
        bundle.detections = detections
        return bundle

    def _draw_overlay(self, bgr, boxes, scores, class_ids, queries):
        """Return a copy of `bgr` with each detection's box + `label score` drawn on
        it, one color per query class. Returns None if drawing fails."""
        try:
            import cv2
            vis = bgr.copy()
            palette = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
                       (255, 0, 255), (255, 255, 0)]
            for i in range(int(boxes.shape[0])):
                x1, y1, x2, y2 = (int(v) for v in boxes[i])
                cid = int(class_ids[i])
                color = palette[cid % len(palette)]
                label = queries[cid] if 0 <= cid < len(queries) else str(cid)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{label} {float(scores[i]):.2f}",
                            (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1, cv2.LINE_AA)
            return vis
        except Exception as e:
            self.get_logger().warn(f'yolo overlay draw failed: {e}')
            return None

    def _publish_annotated(self, channel, vis, header):
        """Publish the annotated BGR frame as a raw sensor_msgs/Image (bgr8),
        carrying the source frame's header so RViz can display it."""
        try:
            img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            img_msg.header = header
            channel.annotated_publisher.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(
                f'{channel.annotated_topic}: publish failed: {e}')

    def _save_overlay_to_disk(self, rec, vis):
        """Save the annotated frame to the model log dir (visualize toggle)."""
        try:
            import cv2
            cv2.imwrite(rec.path('overlay', 'png'), vis)
        except Exception as e:
            self.get_logger().warn(f'yolo overlay save failed: {e}')


def RunYoloServer(args=None):
    rclpy.init(args=args)
    node = YoloServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    RunYoloServer()
