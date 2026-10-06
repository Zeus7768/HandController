"""Passive landmark replay. This module never imports the input layer."""
import json


def frame_record(frame_index, timestamp_ms, hands):
    return {
        "timestamp_ms": int(timestamp_ms),
        "frame": int(frame_index),
        "hands": list(hands),
    }


def hand_record(hand_id, handedness, confidence, raw_points, stable_points):
    def pack(points):
        array = points.reshape(-1, 3)
        return [[float(value) for value in row] for row in array]

    return {
        "hand_id": hand_id,
        "handedness": handedness,
        "confidence": float(confidence),
        "landmarks_raw": pack(raw_points),
        "landmarks_stabilized": pack(stable_points),
    }


def write_frame(handle, record):
    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    handle.flush()


def read_frames(path):
    frames = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            frames.append(json.loads(text))
    return frames
