#!/usr/bin/env python3
"""
sam_server — ROS 2 service node exposing custom_ros_messages/srv/SamSegment.

Promptable segmentation via a self-contained SAM3 wrapper (see SAM3 below).
Accepts a text/concept prompt and/or positive & negative box exemplars and returns
the single best mask (mono8) plus its confidence score.

Note: SAM3's image API supports text + positive/negative BOX prompts only — it has
no click-point prompting — so this service takes box exemplars, not points.
"""
import os
import threading

import numpy as np
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from cv_bridge import CvBridge

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from custom_ros_messages.srv import SamSegment

from model_server.model_logging import ModelLogger, declare_logging_params

# ----- config -----
# SAM3 weights live under <package>/weights/. Set SAM3_MODEL to None to instead
# auto-download from HuggingFace (facebook/sam3, requires HF auth).
SAM3_MODEL = "sam3.pt"
SAM3_CONFIDENCE_THRESHOLD = 0.5

# Weights resolve relative to this package (works with `colcon --symlink-install`,
# which symlinks the installed module back to the source tree). realpath (not
# abspath) is required to follow that symlink. Override the root with
# MODEL_SERVER_WEIGHTS_DIR.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_WEIGHTS_DIR = os.environ.get("MODEL_SERVER_WEIGHTS_DIR",
                              os.path.join(_PKG_ROOT, "weights"))


class SAM3:
    """Minimal SAM3 promptable-segmentation wrapper for the sam_segment service.

    Encodes one image (`_set_image`) then accumulates text + positive/negative box
    prompts for a single query and returns the single best mask (`segment`). SAM3
    has no click-point prompting, so geometric prompts are boxes only.
    """

    def __init__(self):
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))
        if SAM3_MODEL is not None:
            weight_path = os.path.join(_WEIGHTS_DIR, SAM3_MODEL)
            assert os.path.exists(weight_path), f"[SAM3 init] Weights at {weight_path} not found"
            sam_model = build_sam3_image_model(checkpoint_path=weight_path, load_from_HF=False)
        else:
            sam_model = build_sam3_image_model()  # downloads from facebook/sam3 (requires HF auth)
        self.processor = Sam3Processor(sam_model, confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)
        self._inference_state = None
        self._img_hw = None
        print(f"[SAM3 init] Initialized on device {self.device}")

    def _set_image(self, rgb_img):
        """Encode the frame once; the per-prompt decode in segment reuses this."""
        pil_img = PILImage.fromarray(rgb_img)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            self._inference_state = self.processor.set_image(pil_img)
        self._img_hw = (rgb_img.shape[0], rgb_img.shape[1])

    def segment(self, rgb_img, positive_boxes, negative_boxes, text, debug=False):
        """Promptable single-image segmentation.

        Accumulates all prompts for ONE query and returns the single best mask:
          - text:           concept/phrase prompt (Sam3Processor.set_text_prompt)
          - positive_boxes: foreground box exemplars, each [x1, y1, x2, y2] pixel xyxy
          - negative_boxes: background box exemplars, same format
        At least a text prompt or one positive box should be supplied.

        Returns (mask, score): mask is an (H, W) bool numpy array at input
        resolution; returns (None, 0.0) if SAM3 produced no mask.
        """
        self._set_image(rgb_img)                 # encodes frame → self._inference_state, self._img_hw
        state = self._inference_state
        self.processor.reset_all_prompts(state)
        H, W = self._img_hw

        def _norm_cxcywh(box):
            x1, y1, x2, y2 = box
            return [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            if text:
                ret = self.processor.set_text_prompt(prompt=text, state=state)
                if ret is not None:
                    state = ret
            for box in positive_boxes:
                ret = self.processor.add_geometric_prompt(box=_norm_cxcywh(box), label=True, state=state)
                if ret is not None:
                    state = ret
            for box in negative_boxes:
                ret = self.processor.add_geometric_prompt(box=_norm_cxcywh(box), label=False, state=state)
                if ret is not None:
                    state = ret

        masks = state.get("masks")    # (N, 1, H, W) bool
        scores = state.get("scores")  # (N,)
        if masks is None or masks.shape[0] == 0:
            if debug:
                print("[SAM3 segment] no mask produced for the given prompts")
            return None, 0.0
        best = int(scores.argmax().item()) if scores is not None else 0
        mask_np = masks[best, 0].cpu().numpy().astype(bool)
        score = float(scores[best].item()) if scores is not None else 1.0
        if debug:
            print(f"[SAM3 segment] masks={tuple(masks.shape)} best={best} score={score:.3f}")
        return mask_np, score


def _group_boxes(flat):
    """Flattened [x1,y1,x2,y2, ...] → list of [x1,y1,x2,y2]. Raises on a bad length."""
    if len(flat) % 4 != 0:
        raise ValueError(f"box array length {len(flat)} is not a multiple of 4")
    return [[float(v) for v in flat[i:i + 4]] for i in range(0, len(flat), 4)]


# Max time a request waits for the per-image SAM3 model lock before giving up. A
# normal inference is a few seconds; if the lock is held far longer a prior call
# has wedged (CUDA stall/OOM). Failing fast keeps the node responsive (returning a
# clear error) instead of silently queueing every future request on a dead lock.
LOCK_ACQUIRE_TIMEOUT_SEC = 20.0


class SamServer(Node):
    def __init__(self):
        super().__init__("sam_server")
        self.get_logger().info("\n\n\nsam_server beginning SAM3 initialization ...\n\n\n")
        self.bridge = CvBridge()
        log, viz, clear = declare_logging_params(self)
        self.logger = ModelLogger(self, 'sam', 'model_server', __file__,
                                  log=log, visualize=viz, clear=clear)
        # SAM3 keeps a per-image inference state, so concurrent calls must serialize.
        self._lock = threading.RLock()
        self.get_logger().info("sam_server loading SAM3 ...")
        self.sam = SAM3()  # eager load: surfaces weight/GPU errors at startup, not first call
        self.srv = self.create_service(
            SamSegment, "sam_segment", self.segment_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info("sam_server ready")

    def segment_callback(self, request, response):
        rec = self.logger.start()
        rec.set(text=request.text,
                n_positive_boxes=len(request.positive_boxes) // 4,
                n_negative_boxes=len(request.negative_boxes) // 4)
        try:
            positive_boxes = _group_boxes(request.positive_boxes)
            negative_boxes = _group_boxes(request.negative_boxes)
        except ValueError as e:
            return self._fail(response, str(e), rec)
        text = request.text

        if not text and not positive_boxes:
            return self._fail(
                response, "provide a text prompt and/or at least one positive box", rec)

        try:
            rgb = self.bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding="rgb8")
        except Exception as e:
            return self._fail(response, f"failed to decode image: {e}", rec)
        rec.save_image('input', rgb, rgb=True)

        if not self._lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT_SEC):
            return self._fail(
                response,
                f"SAM busy: model lock held > {LOCK_ACQUIRE_TIMEOUT_SEC:.0f}s "
                "(prior inference wedged or GPU overloaded)", rec)
        try:
            mask, score = self.sam.segment(rgb, positive_boxes, negative_boxes, text, debug=False)
        except Exception as e:
            self.get_logger().error(f"SAM3 segment failed: {e}")
            return self._fail(response, f"segmentation error: {e}", rec)
        finally:
            self._lock.release()

        if mask is None:
            return self._fail(response, "SAM3 produced no mask for the given prompts", rec)

        mask_u8 = mask.astype(np.uint8) * 255
        response.mask = self.bridge.cv2_to_imgmsg(mask_u8, encoding="mono8")
        response.mask.header = request.image.header
        response.score = float(score)
        response.success = True
        response.message = "ok"
        # SAM metrics: best-mask confidence + how much of the frame it covers.
        H, W = mask.shape
        area = int(mask.sum())
        coverage = area / float(mask.size) if mask.size else 0.0
        rec.set(score=float(score), mask_pixels=area,
                mask_coverage=round(coverage, 4), image_hw=[int(H), int(W)])
        self.get_logger().info(
            f"SAM metrics: score={score:.3f} mask={area}px "
            f"({coverage * 100:.1f}% of {W}x{H}) text={text!r} "
            f"+boxes={len(positive_boxes)} -boxes={len(negative_boxes)}")
        rec.save_image('mask', mask_u8)
        self._save_sam_overlay(rec, rgb, mask, positive_boxes, negative_boxes)
        rec.finish(success=True, message="ok")
        return response

    def _save_sam_overlay(self, rec, rgb, mask, positive_boxes, negative_boxes):
        """Save the input image with the mask tinted in + the box exemplars drawn
        (green = positive, red = negative). Visualization-only."""
        if not self.logger.visualize:
            return
        try:
            import cv2
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            m = mask.astype(bool)
            tint = np.zeros_like(bgr)
            tint[m] = (0, 255, 0)
            bgr = cv2.addWeighted(bgr, 1.0, tint, 0.5, 0.0)
            for boxes, color in ((positive_boxes, (0, 255, 0)),
                                 (negative_boxes, (0, 0, 255))):
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.imwrite(rec.path('overlay', 'png'), bgr)
        except Exception as e:
            self.get_logger().warn(f'sam overlay failed: {e}')

    def _fail(self, response, msg, rec=None):
        # Leave response.mask as the default (empty) Image.
        response.score = 0.0
        response.success = False
        response.message = msg
        self.get_logger().warn(f"sam_segment: {msg}")
        if rec is not None:
            rec.finish(success=False, message=msg)
        return response


def RunSamServer(args=None):
    rclpy.init(args=args)
    node = SamServer()
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
    RunSamServer()
