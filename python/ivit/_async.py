"""Bounded inference submission for native models.

Returned concurrent.futures.Future objects work with asyncio.wrap_future.
Cancellation applies to queued work; running native inference completes normally.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore

import numpy as np

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ivit-inference")
_slots = BoundedSemaphore(32)  # Includes running and queued requests.


def submit(model, images, config, batch=False):
    if not _slots.acquire(blocking=False):
        raise RuntimeError("SDK async queue is full (32 requests); wait for a Future")
    try:
        # Caller-owned arrays and mutable configuration cannot change queued work.
        snapshot = type(config)()
        for field in (
            "conf_threshold",
            "iou_threshold",
            "max_detections",
            "classes",
            "agnostic_nms",
        ):
            setattr(snapshot, field, getattr(config, field))
        if batch:
            owned = [np.array(image, copy=True, order="C") for image in images]
            method = model.predict_batch
        else:
            owned = np.array(images, copy=True, order="C")
            method = model.predict
        future = _executor.submit(method, owned, snapshot)
    except BaseException:
        _slots.release()
        raise
    future.add_done_callback(lambda _: _slots.release())
    return future
