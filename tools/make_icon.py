"""Build the HandController PNG + ICO with a real alpha channel.

The icon is drawn entirely by this script (no third-party artwork): an open
hand with a cyan-to-blue gradient, a dark outline that stays readable on light
backgrounds, and white landmark points like the MediaPipe hand skeleton.

Shapes are rasterised at SUPERSAMPLE px and downscaled with premultiplied
alpha, which gives anti-aliased edges at every icon size.
"""
from __future__ import annotations

import os
import struct

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
PNG_PATH = os.path.join(ASSETS, "handcontroller_icon.png")
TRANSPARENT_PNG = os.path.join(ASSETS, "handcontroller_icon_transparent.png")
ICO_PATH = os.path.join(ASSETS, "handcontroller_icon.ico")
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
PNG_SIZE = 512
SUPERSAMPLE = 2048
PAD_RATIO = 0.03

# Colours are BGR.
TOP = (238, 211, 34)       # #22D3EE
BOTTOM = (235, 99, 37)     # #2563EB
OUTLINE = (42, 23, 15)     # #0F172A
LANDMARK = (255, 255, 255)

# Geometry in percent of the canvas. Fingers: (base, tip, radius).
PALM = (29, 47, 73, 86, 10)
WRIST = (38, 78, 64, 96, 5)
FINGERS = (
    ((37, 52), (33, 14), 5.6),   # index
    ((47, 50), (47, 8), 5.8),    # middle
    ((57, 52), (60, 13), 5.6),   # ring
    ((66, 57), (73, 24), 5.0),   # pinky
    ((36, 72), (16, 46), 6.0),   # thumb
)
WRIST_POINT = (51, 89)
OUTLINE_WIDTH = 1.6
BONE_WIDTH = 1.3
JOINT_RADIUS = 2.1


def _px(value, size=SUPERSAMPLE):
    return int(round(value * size / 100.0))


def _pt(point, size=SUPERSAMPLE):
    return _px(point[0], size), _px(point[1], size)


def _capsule(mask, a, b, radius, value=255):
    width = max(1, _px(2 * radius))
    cv2.line(mask, _pt(a), _pt(b), value, width, cv2.LINE_8)
    cv2.circle(mask, _pt(a), _px(radius), value, -1, cv2.LINE_8)
    cv2.circle(mask, _pt(b), _px(radius), value, -1, cv2.LINE_8)


def _rounded_rect(mask, rect, value=255):
    x0, y0, x1, y1, radius = rect
    r = _px(radius)
    X0, Y0, X1, Y1 = _px(x0), _px(y0), _px(x1), _px(y1)
    cv2.rectangle(mask, (X0 + r, Y0), (X1 - r, Y1), value, -1)
    cv2.rectangle(mask, (X0, Y0 + r), (X1, Y1 - r), value, -1)
    for cx, cy in ((X0 + r, Y0 + r), (X1 - r, Y0 + r), (X0 + r, Y1 - r), (X1 - r, Y1 - r)):
        cv2.circle(mask, (cx, cy), r, value, -1, cv2.LINE_8)


def _lerp(a, b, t):
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def hand_mask(size=SUPERSAMPLE):
    mask = np.zeros((size, size), np.uint8)
    _rounded_rect(mask, PALM)
    _rounded_rect(mask, WRIST)
    for base, tip, radius in FINGERS:
        _capsule(mask, base, tip, radius)
    return mask


def landmark_mask(size=SUPERSAMPLE):
    """Skeleton: wrist -> finger base -> two joints -> tip, like MediaPipe's 21 points."""
    mask = np.zeros((size, size), np.uint8)
    bone = max(1, _px(BONE_WIDTH))
    joints = [WRIST_POINT]
    bases = []
    for base, tip, radius in FINGERS:
        dx, dy = tip[0] - base[0], tip[1] - base[1]
        length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        end = (tip[0] - dx / length * radius * 0.55, tip[1] - dy / length * radius * 0.55)
        chain = [base, _lerp(base, end, 0.45), _lerp(base, end, 0.75), end]
        for a, b in zip(chain, chain[1:]):
            cv2.line(mask, _pt(a), _pt(b), 255, bone, cv2.LINE_8)
        cv2.line(mask, _pt(WRIST_POINT), _pt(base), 255, bone, cv2.LINE_8)
        joints.extend(chain)
        bases.append(base)
    for a, b in zip(bases[:4], bases[1:4]):
        cv2.line(mask, _pt(a), _pt(b), 255, bone, cv2.LINE_8)
    for point in joints:
        cv2.circle(mask, _pt(point), _px(JOINT_RADIUS), 255, -1, cv2.LINE_8)
    return mask


def draw_icon(size=SUPERSAMPLE):
    """Full-resolution BGRA artwork, transparent outside the outlined hand."""
    inside = hand_mask(size)
    kernel_size = max(3, _px(OUTLINE_WIDTH) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    silhouette = cv2.dilate(inside, kernel)
    t = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    gradient = np.array(TOP, np.float32) * (1.0 - t) + np.array(BOTTOM, np.float32) * t
    image = np.empty((size, size, 3), np.float32)
    image[:] = np.array(OUTLINE, np.float32)
    fill = inside > 0
    image[fill] = np.broadcast_to(gradient, (size, size, 3))[fill]
    bones = (landmark_mask(size) > 0) & fill
    image[bones] = np.array(LANDMARK, np.float32)
    bgra = np.dstack([image, silhouette.astype(np.float32)])
    bgra = np.clip(np.rint(bgra), 0, 255).astype(np.uint8)
    bgra[bgra[:, :, 3] == 0, :3] = 0
    return bgra


def crop_to_alpha(bgra, pad_ratio=PAD_RATIO):
    """Square, centred, padded crop of the opaque content. Proportions are kept."""
    alpha = bgra[:, :, 3]
    rows = np.where(alpha.max(axis=1) > 8)[0]
    cols = np.where(alpha.max(axis=0) > 8)[0]
    if rows.size == 0 or cols.size == 0:
        return bgra
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    crop = bgra[y0:y1, x0:x1]
    side = max(crop.shape[0], crop.shape[1])
    pad = max(8, int(side * pad_ratio))
    canvas = np.zeros((side + 2 * pad, side + 2 * pad, 4), dtype=np.uint8)
    oy = pad + (side - crop.shape[0]) // 2
    ox = pad + (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return canvas


def resize_bgra(bgra, size):
    """Premultiplied-alpha area downscale: no dark or coloured halo on the edges.
    Sizes up to 48 px get a light unsharp mask so details stay readable."""
    work = bgra.astype(np.float32) / 255.0
    alpha = work[:, :, 3:4]
    premul = np.dstack([work[:, :, :3] * alpha, alpha])
    interpolation = cv2.INTER_AREA if size <= min(bgra.shape[:2]) else cv2.INTER_LANCZOS4
    small = cv2.resize(premul, (size, size), interpolation=interpolation)
    small = np.clip(small, 0.0, 1.0)
    a = small[:, :, 3:4]
    rgb = np.where(a > 1e-4, small[:, :, :3] / np.maximum(a, 1e-4), 0.0)
    if size <= 48:
        blur = cv2.GaussianBlur(rgb, (0, 0), 0.6)
        rgb = np.clip(rgb + 0.6 * (rgb - blur), 0.0, 1.0)
    out = np.dstack([rgb, a])
    out = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    out[out[:, :, 3] == 0, :3] = 0
    return out


def encode_png(bgra):
    ok, blob = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG encode failed")
    return blob.tobytes()


def write_png(path, bgra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(encode_png(bgra))


def write_ico(path, master_bgra, sizes=ICO_SIZES):
    blobs = [encode_png(resize_bgra(master_bgra, size)) for size in sizes]
    offset = 6 + 16 * len(blobs)
    directory = []
    for size, blob in zip(sizes, blobs):
        edge = 0 if size >= 256 else size
        directory.append(struct.pack("<BBBBHHII", edge, edge, 0, 0, 1, 32, len(blob), offset))
        offset += len(blob)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<HHH", 0, 1, len(blobs)))
        handle.write(b"".join(directory))
        for blob in blobs:
            handle.write(blob)


def build():
    os.makedirs(ASSETS, exist_ok=True)
    square = crop_to_alpha(draw_icon())
    # Every size is rendered from the full-resolution square, never from a small master.
    master = resize_bgra(square, PNG_SIZE)
    write_png(PNG_PATH, master)
    write_png(TRANSPARENT_PNG, master)
    write_ico(ICO_PATH, square)
    alpha = master[:, :, 3]
    return {
        "png": PNG_PATH,
        "png_transparent": TRANSPARENT_PNG,
        "ico": ICO_PATH,
        "size": master.shape[:2],
        "opaque_pixels": int(np.count_nonzero(alpha)),
        "transparent_pixels": int(np.count_nonzero(alpha == 0)),
        "has_partial_alpha": bool(np.any((alpha > 0) & (alpha < 255))),
    }


if __name__ == "__main__":
    for key, value in build().items():
        print("%s=%s" % (key, value))
