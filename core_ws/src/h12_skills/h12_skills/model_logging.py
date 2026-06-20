#!/usr/bin/env python3
"""Per-model debug logging + visualization for the perception/skill nodes.

A small, dependency-light helper each model node (graspgen / gemini / sam /
skills) instantiates from two ROS parameters:

    enable_logging        save raw inputs/outputs + a per-call metadata record
    enable_visualization  save annotated overlay images
    clear_logs            wipe this model's log dir on startup (default true)

Output goes to a package-relative tree so it persists on the host through the
core_ws bind mount and survives container restarts:

    <package>/logs/<model_name>/
        calls.jsonl                  one JSON line per call (when enabled)
        <id>_<tag>.{png,npy,txt}     per-call artifacts

<id> is "<YYYYmmdd_HHMMSS_mmm>_<seq>" so artifacts sort chronologically and
group by call. Set MODEL_LOG_DIR to override the output root (mainly for tests).

This module is intentionally duplicated verbatim in h12_skills and
vision_pipeline — they are independent ROS packages with no shared dependency.
Keep the two copies in sync.
"""

import json
import os
import shutil
import threading
import time
from datetime import datetime

import numpy as np


def _log_root(pkg_name, anchor_file):
    """Resolve "<package>/logs" for pkg_name (the per-model subdir is appended by
    ModelLogger).

    Order: MODEL_LOG_DIR override -> the bind-mounted source tree
    (/home/code/core_ws/src/<pkg>) -> walk up from anchor_file to the dir
    holding package.xml -> cwd. The source-tree branch is what makes output
    persist on the host (the nodes run from build/, which a clean rebuild wipes).
    """
    override = os.environ.get('MODEL_LOG_DIR')
    if override:
        return override
    src = os.path.join('/home/code/core_ws/src', pkg_name)
    if os.path.isdir(src):
        return os.path.join(src, 'logs')
    d = os.path.dirname(os.path.realpath(anchor_file))
    for _ in range(8):
        if os.path.isfile(os.path.join(d, 'package.xml')):
            return os.path.join(d, 'logs')
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.join(os.getcwd(), 'logs')


def _imwrite(path, img, rgb):
    """Write an image with cv2 (lazy import). img is BGR unless rgb=True."""
    import cv2
    img = np.asarray(img)
    if rgb and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


class _Call:
    """One model invocation's artifact sink. Every method is gated by the parent
    logger's flags, so node code can call them unconditionally."""

    def __init__(self, logger, seq):
        self._logger = logger
        self._t0 = time.monotonic()
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        self.id = f'{stamp}_{seq:04d}'
        self.meta = {'id': self.id, 'model': logger.model_name}

    def path(self, tag, ext):
        """Absolute path for an artifact of this call (caller writes it itself)."""
        return os.path.join(self._logger.dir, f'{self.id}_{tag}.{ext}')

    def set(self, **fields):
        """Merge metadata fields into this call's JSONL record."""
        self.meta.update(fields)
        return self

    # --- raw artifacts: gated on enable_logging --------------------------------
    def save_image(self, tag, img, rgb=False):
        if self._logger.log:
            self._safe(lambda: _imwrite(self.path(tag, 'png'), img, rgb), tag)

    def save_array(self, tag, arr):
        if self._logger.log:
            self._safe(lambda: np.save(self.path(tag, 'npy'), np.asarray(arr)), tag)

    def save_text(self, tag, text):
        if self._logger.log:
            def _w():
                with open(self.path(tag, 'txt'), 'w') as f:
                    f.write('' if text is None else str(text))
            self._safe(_w, tag)

    # --- annotated overlays: gated on enable_visualization ---------------------
    def save_overlay(self, tag, img, rgb=False):
        if self._logger.visualize:
            self._safe(lambda: _imwrite(self.path(tag, 'png'), img, rgb), tag)

    def _safe(self, fn, tag):
        try:
            fn()
        except Exception as e:  # logging must never take down the service
            self._logger._warn(f'failed to write {tag}: {e}')

    def finish(self, success=None, message=None):
        """Append the metadata record (latency auto-filled). Call once per call."""
        if success is not None:
            self.meta['success'] = bool(success)
        if message is not None:
            self.meta['message'] = str(message)
        self.meta['latency_ms'] = round((time.monotonic() - self._t0) * 1000.0, 1)
        try:
            line = json.dumps(self.meta, default=str)
            with self._logger._lock:
                with open(self._logger._meta_path, 'a') as f:
                    f.write(line + '\n')
        except Exception as e:
            self._logger._warn(f'failed to append metadata: {e}')


class _NullCall:
    """No-op call returned when both toggles are off — keeps node code branch-free."""
    id = ''

    def path(self, *a, **k):
        return os.devnull

    def set(self, **_):
        return self

    def save_image(self, *a, **k):
        pass

    def save_array(self, *a, **k):
        pass

    def save_text(self, *a, **k):
        pass

    def save_overlay(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass


class ModelLogger:
    """Owns the per-model output dir and the two toggles. Construct from ROS
    params; call start() once per request to get a per-call artifact sink."""

    def __init__(self, node, model_name, pkg_name, anchor_file,
                 log=False, visualize=False, clear=False):
        self._node = node
        self.model_name = model_name
        self.log = bool(log)
        self.visualize = bool(visualize)
        self.dir = os.path.join(_log_root(pkg_name, anchor_file), model_name)
        self._meta_path = os.path.join(self.dir, 'calls.jsonl')
        self._lock = threading.Lock()
        self._seq = 0
        # Wipe this model's log dir on startup so each run begins fresh (independent
        # of the log/viz toggles — clears stale artifacts even when output is off).
        if clear:
            self._clear_dir()
        if self.enabled:
            try:
                os.makedirs(self.dir, exist_ok=True)
                self._info(f'logging={self.log} visualization={self.visualize} '
                           f'-> {self.dir}')
            except OSError as e:
                self._warn(f'cannot create {self.dir}: {e}; disabling output')
                self.log = self.visualize = False

    def _clear_dir(self):
        """Remove any existing contents of this model's log dir (best-effort)."""
        try:
            if os.path.isdir(self.dir):
                shutil.rmtree(self.dir)
                self._info(f'cleared previous logs in {self.dir}')
        except OSError as e:
            self._warn(f'could not clear {self.dir}: {e}')

    @property
    def enabled(self):
        return self.log or self.visualize

    def start(self):
        """Begin a call. Returns a sink; a no-op sink when both toggles are off."""
        if not self.enabled:
            return _NullCall()
        with self._lock:
            self._seq += 1
            seq = self._seq
        return _Call(self, seq)

    def _info(self, msg):
        self._node.get_logger().info(f'[{self.model_name} log] {msg}')

    def _warn(self, msg):
        self._node.get_logger().warn(f'[{self.model_name} log] {msg}')


def declare_logging_params(node, *, log_default=False, viz_default=False,
                           clear_default=True):
    """Declare the standard toggles on `node` and return (log, visualize, clear).
    `clear_logs` defaults True: the model's log dir is wiped on startup."""
    log = node.declare_parameter('enable_logging', log_default).value
    viz = node.declare_parameter('enable_visualization', viz_default).value
    clear = node.declare_parameter('clear_logs', clear_default).value
    return bool(log), bool(viz), bool(clear)
