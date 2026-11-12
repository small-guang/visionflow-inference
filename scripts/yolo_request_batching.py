"""CPU-only request validation, bounded batch planning and public output splitting.

No request or image is retained beyond the caller's execute invocation. A plan
contains views of existing transport buffers; only the active group is packed.
"""
from dataclasses import dataclass

import numpy as np


MAX_BATCH_SIZE = 4
MAX_PIXELS = 8_388_608
MAX_PACK_BYTES = MAX_BATCH_SIZE * MAX_PIXELS * 3


class InputValidationError(ValueError):
    pass


@dataclass(frozen=True)
class RequestImage:
    index: int
    images: np.ndarray

    @property
    def batch_size(self):
        return self.images.shape[0]

    @property
    def shape_key(self):
        return self.images.shape[1:]


def validate_images(images, *, max_batch_size=MAX_BATCH_SIZE, imgsz=640):
    if not isinstance(images, np.ndarray):
        raise InputValidationError("RAW_IMAGE must be a numpy array")
    if images.ndim != 4 or images.shape[3] != 3:
        raise InputValidationError("RAW_IMAGE batch must have shape [B,H,W,3]")
    if images.dtype != np.uint8:
        raise InputValidationError("RAW_IMAGE must have dtype uint8")
    if not 1 <= images.shape[0] <= min(max_batch_size, MAX_BATCH_SIZE):
        raise InputValidationError(f"RAW_IMAGE batch size must be between 1 and {max_batch_size}")
    _, height, width, _ = images.shape
    if height <= 0 or width <= 0:
        raise InputValidationError("RAW_IMAGE height and width must be positive")
    if max(height, width) > 8192:
        raise InputValidationError("RAW_IMAGE exceeds max side 8192")
    if height * width > MAX_PIXELS:
        raise InputValidationError("RAW_IMAGE exceeds max pixels 8388608")
    ratio = min(imgsz / height, imgsz / width)
    if min(round(width * ratio), round(height * ratio)) < 1:
        raise InputValidationError("RAW_IMAGE aspect ratio produces a resize dimension smaller than 1 pixel")


def plan_groups(items, *, combine=True):
    """Stable same-shape groups, never splitting an individual RPC's batch."""
    groups, pending = [], {}
    for item in items:
        if not combine:
            groups.append([item])
            continue
        key = item.shape_key
        group = pending.get(key)
        if group is None or sum(member.batch_size for member in group) + item.batch_size > MAX_BATCH_SIZE:
            group = []
            groups.append(group)
            pending[key] = group
        group.append(item)
    return groups


def pack_group(group):
    """At most 96 MiB extra host storage; a single RPC remains zero-copy."""
    if not group:
        raise ValueError("cannot pack an empty request group")
    batch_size = sum(item.batch_size for item in group)
    if (batch_size > MAX_BATCH_SIZE or any(item.shape_key != group[0].shape_key for item in group)
            or sum(item.images.nbytes for item in group) > MAX_PACK_BYTES):
        raise ValueError("request group exceeds shape, batch or host memory budget")
    if len(group) == 1:
        return group[0].images, 0
    images = np.concatenate([item.images for item in group], axis=0)
    return images, images.nbytes


def split_outputs(values, group, *, segmentation=False):
    """Yield (request index, owned contiguous outputs, copied host bytes)."""
    batch_size = sum(item.batch_size for item in group)
    if len(values) not in (2, 3) or any(value.shape[0] != batch_size for value in values):
        raise RuntimeError("YOLO output batch does not match the request group")
    offset = 0
    for item in group:
        stop = offset + item.batch_size
        slices = [value[offset:stop] for value in values]
        if segmentation:
            counts = slices[2]
            if counts.shape != (item.batch_size, 1) or np.any(counts < 0):
                raise RuntimeError("invalid segmentation detection counts")
            slots = max(1, int(counts.max()))
            if slots > slices[0].shape[1] or slots > slices[1].shape[1]:
                raise RuntimeError("segmentation count exceeds output capacity")
            slices[0], slices[1] = slices[0][:, :slots], slices[1][:, :slots]
        # Single-RPC outputs from cp.asnumpy already own their memory. Reuse
        # them when the original canonical shape matches, otherwise detach.
        if len(group) == 1 and all(view.shape == value.shape and value.flags.c_contiguous
                                   and value.flags.owndata for view, value in zip(slices, values)):
            outputs, copied = values, 0
        else:
            outputs = tuple(np.array(value, copy=True, order="C") for value in slices)
            copied = sum(value.nbytes for value in outputs)
        yield item.index, outputs, copied
        offset = stop
