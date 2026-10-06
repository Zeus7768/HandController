"""Editor helpers. They call the existing engine and mapping.

Destructive calls copy gestures.json and mapping.json next to the files.
gestures_backup.json is never opened.
"""
import os
import shutil
import time

import numpy as np

from gesture_engine import FEATURE_SIZE, GestureSamples, gesture_number, name_of, samples_of
import geometry_engine as geo


def format_gesture_label(gesture_id, name=None):
    """Stable id plus optional user name. Never repeats the same field."""
    gid = str(gesture_id or "").strip()
    shown = str(name or "").strip()
    if not gid:
        return shown
    if not shown or shown == gid:
        return gid
    return f"{gid} — {shown}"


def gesture_rows(engine, mapping, side, dynamic_names=None):
    """One row per learned gesture that has a name or samples, plus geometry."""
    rows = []
    for gesture in engine.editable_ids(side):
        entry = engine.database["gestures"][side].get(gesture)
        count = len(list(samples_of(entry)))
        label = name_of(entry)
        if count == 0 and not label:
            continue
        mapped = mapping.get(side, gesture) or {}
        rows.append({
            "id": gesture,
            "name": label,
            "kind": "STATIC",
            "hand": side,
            "enabled": mapped.get("enabled", True) is not False and mapped.get("type") != "NONE",
            "samples": count,
            "action": mapped.get("type", "NONE"),
        })
    for gesture in geo.SPECIAL_BY_SIDE.get(side, ()):
        mapped = mapping.get(side, gesture) or {}
        rows.append({
            "id": gesture,
            "name": gesture,
            "kind": "GEOMETRIC",
            "hand": side,
            "enabled": mapped.get("enabled", True) is not False and mapped.get("type") != "NONE",
            "samples": 0,
            "action": mapped.get("type", "NONE"),
        })
    return rows


def duplicate_gesture(engine, mapping, side, gesture):
    """Copy samples and the action onto a new id. Lists are copied, not shared."""
    source = engine.database["gestures"][side].get(gesture)
    if gesture_number(gesture) is None or source is None:
        return None
    label = name_of(source) or gesture
    new_id = engine.allocate_gesture(side, f"{label} 2")
    if not new_id:
        return None
    copied = [list(sample) for sample in samples_of(source)]
    engine.database["gestures"][side][new_id] = GestureSamples(copied, f"{label} 2")
    engine.save_database()
    engine.rebuild_index()
    mapped = mapping.get(side, gesture)
    if mapped:
        mapping._store(side, new_id, dict(mapped))
    return new_id


def outlier_indices(samples, z_limit=3.5):
    """Indices unusually far from the median sample. Never deletes anything."""
    rows = []
    for sample in samples_of(samples) if not isinstance(samples, list) else samples:
        array = np.asarray(sample, dtype=np.float64).ravel()
        if array.size == FEATURE_SIZE and np.isfinite(array).all():
            rows.append(array)
    if len(rows) < 4:
        return []
    stack = np.stack(rows)
    center = np.median(stack, axis=0)
    distance = np.linalg.norm(stack - center, axis=1)
    med = float(np.median(distance))
    mad = float(np.median(np.abs(distance - med))) or 1e-6
    score = 0.6745 * (distance - med) / mad
    return [index for index, value in enumerate(score) if value > z_limit]


def remove_samples(engine, side, gesture, indices):
    current = engine.database["gestures"][side].get(gesture)
    kept_name = name_of(current)
    samples = [list(sample) for index, sample in enumerate(samples_of(current)) if index not in set(indices)]
    engine.database["gestures"][side][gesture] = GestureSamples(samples, kept_name)
    engine.save_database()
    engine.rebuild_index()
    return len(samples)


def toggle_enabled(mapping, side, gesture):
    entry = mapping.get(side, gesture)
    if not entry:
        return False
    updated = dict(entry)
    updated["enabled"] = not entry.get("enabled", True)
    if updated["enabled"]:
        updated.pop("enabled", None)
    return mapping._store(side, gesture, updated)


def backup_config(gestures_path, mapping_path):
    """Copy the unique config files. Does not touch gestures_backup.json."""
    sources = [path for path in (gestures_path, mapping_path) if path and os.path.isfile(path)]
    if not sources:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = os.path.dirname(os.path.abspath(sources[0]))
    dest = os.path.join(root, "_editor_backups", stamp)
    os.makedirs(dest, exist_ok=True)
    for path in sources:
        shutil.copy2(path, os.path.join(dest, os.path.basename(path)))
    return dest
