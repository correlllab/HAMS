#!/usr/bin/env python3
"""
gemini_server — ROS 2 service node exposing custom_ros_messages/srv/GeminiQuery.

Send an image and/or a text prompt; get Gemini's free-form text answer back.

Uses an env-var API key (GEMINI_API_KEY / GOOGLE_API_KEY), the GEMINI_MODEL
constant below, and a GEMINI_MAX_RETRIES retry loop, with a general
spatial-reasoning system prompt (SYSTEM_INSTRUCTION) and no forced JSON mime type,
so the caller's prompt chooses the output format (plain text, JSON, coordinates).
"""
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from cv_bridge import CvBridge
from google import genai
from google.genai import types
from PIL import Image

from custom_ros_messages.srv import GeminiQuery

from model_server.model_logging import ModelLogger, declare_logging_params

# ----- config -----
GEMINI_MODEL = "gemini-robotics-er-1.6-preview"
GEMINI_MAX_RETRIES = 4
# Per-attempt HTTP timeout in MILLISECONDS (google-genai expects ms). The SDK
# sets NO timeout by default and passes timeout=None straight to httpx, so a
# stalled/half-open socket makes generate_content() block FOREVER — the retry
# loop below never fires because a hang is not an exception. With this set, a
# stall raises after the deadline and the loop can retry / fail cleanly.
# 210 s sits just under the skills client's 240 s GEMINI_TIMEOUT_SEC, so the
# first attempt gets the full budget before that ceiling fires (a retry would
# push past it, so under the live skill expect ~one full-length attempt).
GEMINI_HTTP_TIMEOUT_MS = 210_000
# "Thinking" (reasoning) token budget for Robotics-ER. UNSET defaults to dynamic
# (-1): the model self-allocates a variable amount of reasoning per request, which
# is the main reason latency swings wildly (a few seconds to >170 s on the same
# detection). 0 DISABLES thinking — Google's Robotics-ER docs use thinking_budget=0
# for object detection / pointing / bounding boxes (our use), giving the lowest,
# most consistent latency. Raise it (e.g. 1024) or set -1 only if a caller needs
# complex spatial reasoning (counting, gauge reading) at the cost of latency.
GEMINI_THINKING_BUDGET = 0


# Robust, task-agnostic system prompt for the shared gemini_query service. It
# pins output discipline (obey the caller's format; JSON-only when JSON is asked,
# no fences) and the image-coordinate convention (normalized 0-1000, y-first,
# box_2d=[y1,x1,y2,x2]) so downstream parsers — e.g. the grasp skill's box parse
# — are reliable, without forcing any one output format on every caller.
SYSTEM_INSTRUCTION = """\
You are a precise visual-perception and spatial-reasoning assistant for a humanoid robot. \
You are given a camera image together with a question or instruction, \
and you answer strictly from what is visible in that image.

Follow these rules:
- Ground every answer in the image. Never invent objects, parts, or attributes \
that are not visible. If the requested thing is not present, say so plainly (or \
return an empty result in the requested format) instead of guessing a location.
- Obey the caller's requested output format EXACTLY. If the prompt asks for JSON, \
reply with only valid, parseable JSON — no markdown code fences, no comments, no \
trailing text. If it asks for a single value or word, return only that.
- Use the standard image-coordinate convention whenever coordinates are requested \
and the prompt does not specify another: values are normalized to 0-1000 over the \
image with the origin at the TOP-LEFT. A point is [y, x] (row first, then column). \
A bounding box is box_2d = [y_min, x_min, y_max, x_max] (top-left then \
bottom-right).
- Be decisive and literal. Prefer a best-guess answer over refusing or hedging; \
add no disclaimers, apologies, or explanations unless explicitly asked. When \
uncertain, still return your single most-likely answer (with a low confidence \
score if the requested format includes one).
- Keep responses minimal: return only what was asked for, and nothing else.
"""


def _wrap(text, width):
    """Wrap text to lines of at most `width` chars for the overlay caption."""
    import textwrap
    text = (text or "").replace("\n", " ")
    return textwrap.wrap(text, width) or [""]


def _extract_boxes(text):
    """Best-effort: pull "box_2d": [y1,x1,y2,x2] boxes (normalized 0-1000, y first)
    out of a Gemini JSON answer. Strips ```json fences and tolerates a bare object
    or array. Returns a list of [y1,x1,y2,x2] (empty when none/unparseable)."""
    import json
    import re
    if not text:
        return []
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    payload = m.group(1) if m else text
    data = None
    try:
        data = json.loads(payload)
    except Exception:
        a, b = payload.find("["), payload.rfind("]")
        if 0 <= a < b:
            try:
                data = json.loads(payload[a:b + 1])
            except Exception:
                data = None
    if data is None:
        return []
    entries = data if isinstance(data, list) else [data]
    boxes = []
    for e in entries:
        if isinstance(e, dict) and "box_2d" in e:
            try:
                y1, x1, y2, x2 = (float(v) for v in e["box_2d"])
                boxes.append([y1, x1, y2, x2])
            except (ValueError, TypeError):
                continue
    return boxes


class GeminiServer(Node):
    def __init__(self):
        super().__init__("gemini_server")
        self.bridge = CvBridge()
        log, viz, clear = declare_logging_params(self)
        self.logger = ModelLogger(self, 'gemini', 'model_server', __file__,
                                  log=log, visualize=viz, clear=clear)

        # Model from the constant above; API key from the environment.
        self.model = GEMINI_MODEL
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set — export it before running gemini_server "
                "(get a key at https://aistudio.google.com/apikey)")
        # http_options.timeout (ms) bounds each call so a stalled network raises
        # instead of hanging generate_content() indefinitely — see the constant.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=GEMINI_HTTP_TIMEOUT_MS),
        )

        # General-purpose spatial-reasoning config: a robust system_instruction
        # (format obedience + the 0-1000 y-first coordinate convention) but no
        # forced response_mime_type — callers choose their own output format
        # (plain text, JSON, ...) in their prompt.
        self.gen_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
            ],
            temperature=0.1,
            # Pin the reasoning budget (see GEMINI_THINKING_BUDGET) so detection
            # latency is low and consistent instead of dynamically variable.
            thinking_config=types.ThinkingConfig(
                thinking_budget=GEMINI_THINKING_BUDGET),
        )
        self.max_retries = GEMINI_MAX_RETRIES

        # Network-bound + stateless per call, so a reentrant group + multithreaded
        # executor lets concurrent requests run without serializing.
        self.srv = self.create_service(
            GeminiQuery, "gemini_query", self.query_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(f"gemini_server ready (model={self.model})")

    def query_callback(self, request, response):
        t_start = time.monotonic()
        rec = self.logger.start()
        rec.set(prompt=request.prompt, has_image=len(request.image.data) > 0,
                model=self.model)
        rec.save_text('prompt', request.prompt)

        has_image = len(request.image.data) > 0
        preview = (request.prompt or "")[:80].replace("\n", " ")
        self.get_logger().info(
            f"request received: image={has_image} prompt={preview!r} "
            f"(model={self.model})")

        # Build the Gemini contents list from whichever inputs were provided.
        contents = []
        rgb = None
        if has_image:
            try:
                rgb = self.bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding="rgb8")
                contents.append(Image.fromarray(rgb))
                rec.save_image('input', rgb, rgb=True)
            except Exception as e:
                response.response = ""
                response.success = False
                response.message = f"failed to decode image: {e}"
                self.get_logger().error(
                    f"{response.message} (after {time.monotonic() - t_start:.2f}s)")
                rec.finish(success=False, message=response.message)
                return response
        if request.prompt:
            contents.append(request.prompt)

        if not contents:
            response.response = ""
            response.success = False
            response.message = "provide an image and/or a prompt"
            self.get_logger().warn(response.message)
            rec.finish(success=False, message=response.message)
            return response

        # Retry loop. Gemini calls are slow (the grasp model can take minutes), so
        # log when each attempt starts and how long it waited — operators otherwise
        # can't tell a slow call from a hung one.
        errors = []
        while len(errors) < self.max_retries:
            attempt = len(errors) + 1
            self.get_logger().info(
                f"calling Gemini (attempt {attempt}/{self.max_retries}) — waiting "
                f"for response...")
            t_call = time.monotonic()
            try:
                raw = self.client.models.generate_content(
                    model=self.model, contents=contents, config=self.gen_config,
                )
                wait = time.monotonic() - t_call
                response.response = raw.text or ""
                response.success = True
                response.message = "ok"
                rec.set(n_retries=len(errors), wait_sec=round(wait, 2))
                rec.save_text('response', response.response)
                boxes = _extract_boxes(response.response)
                if boxes:
                    rec.set(boxes=boxes)
                self._save_gemini_overlay(rec, rgb, request.prompt,
                                          response.response, boxes)
                rec.finish(success=True, message="ok")
                self.get_logger().info(
                    f"Gemini responded in {wait:.2f}s ({len(boxes)} box(es), "
                    f"{len(response.response)} chars); total "
                    f"{time.monotonic() - t_start:.2f}s")
                # Print the model output itself so operators see what Gemini
                # actually answered (not just timing/length).
                self.get_logger().info(f"Gemini response: {response.response}")
                return response
            except Exception as e:
                wait = time.monotonic() - t_call
                errors.append(str(e))
                self.get_logger().warn(
                    f"Gemini attempt {attempt} failed after {wait:.2f}s: {e}")

        response.response = ""
        response.success = False
        response.message = f"gemini failed after {len(errors)} tries: {errors}"
        rec.set(n_retries=len(errors), errors=errors)
        rec.finish(success=False, message=response.message)
        self.get_logger().error(
            f"Gemini failed after {len(errors)} tries / "
            f"{time.monotonic() - t_start:.2f}s")
        return response

    def _save_gemini_overlay(self, rec, rgb, prompt, answer, boxes=()):
        """Save the input image with any returned bounding box(es) drawn on it and
        the prompt + answer as a caption banner. `boxes` are [y1,x1,y2,x2]
        normalized to 0-1000 (y first). Visualization-only; skipped without an image."""
        if not self.logger.visualize or rgb is None:
            return
        try:
            import cv2
            import numpy as np
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            h, w = bgr.shape[:2]
            for y1, x1, y2, x2 in boxes:
                p1 = (int(x1 / 1000.0 * w), int(y1 / 1000.0 * h))
                p2 = (int(x2 / 1000.0 * w), int(y2 / 1000.0 * h))
                cv2.rectangle(bgr, p1, p2, (0, 255, 0), 2)
            lines = _wrap(f"Q: {prompt}", 70) + _wrap(f"A: {answer}", 70)
            banner = np.full((18 * len(lines) + 12, w, 3), 32, np.uint8)
            for i, ln in enumerate(lines):
                cv2.putText(banner, ln, (6, 18 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1, cv2.LINE_AA)
            out = np.vstack([bgr, banner])
            cv2.imwrite(rec.path('overlay', 'png'), out)
        except Exception as e:
            self.get_logger().warn(f'gemini overlay failed: {e}')


def RunGeminiServer(args=None):
    rclpy.init(args=args)
    node = GeminiServer()
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
    RunGeminiServer()
