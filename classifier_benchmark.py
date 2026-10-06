"""Leave-one-out ML benchmark on gestures.json. Never loads a readonly backup dataset.

Each sample is held out, models are trained on the rest, then that sample is
predicted. This avoids testing on a vector that was in the training set.

SVM and Random Forest need sklearn and at least two distinct gesture IDs.
k-NN uses the same palm-local 63-d features as the live GestureEngine.
"""
import os
import time

import numpy as np

from config import GESTURES_FILE, user_data_path
from ensemble_classifier import EnsembleClassifier
from feature_extractor import shape_features
from gesture_engine import SIDES, UNKNOWN, GestureEngine, samples_of
from ml_engine import MIN_CLASSES, SKIP_INSUFFICIENT, sklearn_available
from ml_features import FEATURE_SIZE, FEATURE_VERSION_BASE, FEATURE_VERSION_EXTENDED

FEATURE_VERSIONS = (FEATURE_VERSION_BASE, FEATURE_VERSION_EXTENDED)


def _load_engine(path=None):
    path = path or user_data_path(GESTURES_FILE)
    if not path or not os.path.isfile(path):
        return None
    if path and os.path.basename(path).lower().endswith("_backup.json"):
        raise ValueError("readonly backup datasets are not allowed")
    return GestureEngine(path)


def bank_vectors(engine, side, feature_version=FEATURE_VERSION_BASE):
    """List of (vector, label). Live path is 63 coords; extended adds 10 distances."""
    side = str(side or "").upper()
    rows = []
    if engine is None or side not in SIDES:
        return rows
    extended = feature_version == FEATURE_VERSION_EXTENDED
    for name, entry in engine.database["gestures"][side].items():
        for sample in samples_of(entry):
            feat = GestureEngine.canonicalize_features(sample)
            if feat is None:
                continue
            if extended:
                local = feat.reshape(21, 3)
                feat = np.concatenate((feat, shape_features(local)))
            rows.append((feat, name))
    return rows


def _knn_predict(train_x, train_y, query, k):
    if train_x.size == 0:
        return None
    delta = train_x - query
    dist = np.einsum("ij,ij->i", delta, delta)
    k = min(max(1, int(k)), dist.size)
    nearest = np.argpartition(dist, k - 1)[:k]
    votes = {}
    for index in nearest:
        label = train_y[int(index)]
        votes[label] = votes.get(label, 0.0) + 1.0 / (float(np.sqrt(dist[int(index)])) + 0.0001)
    return max(votes, key=votes.get)


def _empty_model(reason="", samples=0, classes=0):
    return {
        "samples": int(samples),
        "classes": int(classes),
        "accuracy": None,
        "errors": None,
        "unknown": 0,
        "ambiguous": 0,
        "train_ms": 0.0,
        "predict_avg_ms": 0.0,
        "skipped": reason,
        "disagreements": 0,
    }


def leave_one_out_knn(rows, k=5, use_shape=False, shape_weight=0.35):
    """Accuracy of 1-vs-rest k-NN. Needs at least two samples."""
    if len(rows) < 2:
        return None
    if isinstance(rows[0], tuple) and len(rows[0]) == 3:
        coords = np.stack([row[0] for row in rows], axis=0)
        extras = np.stack([row[1] for row in rows], axis=0)
        labels = [row[2] for row in rows]
        if use_shape:
            scale = max(float(np.std(coords)), 1e-6)
            extra_scale = max(float(np.std(extras)), 1e-6)
            data = np.concatenate((coords / scale, (shape_weight * extras) / extra_scale), axis=1)
        else:
            data = coords
    else:
        data = np.stack([row[0] for row in rows], axis=0)
        labels = [row[1] for row in rows]
    hits = 0
    for index in range(len(rows)):
        mask = np.ones(len(rows), dtype=bool)
        mask[index] = False
        predicted = _knn_predict(data[mask], [labels[i] for i, keep in enumerate(mask) if keep], data[index], k)
        if predicted == labels[index]:
            hits += 1
    return hits / float(len(rows))


def compare_k_values(engine=None, ks=(3, 5, 7)):
    engine = engine or _load_engine()
    scores = {}
    for k in ks:
        parts = []
        for side in SIDES:
            acc = leave_one_out_knn(bank_vectors(engine, side), k=k)
            if acc is not None:
                parts.append(acc)
        scores[int(k)] = float(sum(parts) / len(parts)) if parts else None
    return scores


def compare_shape_features(engine=None, k=5):
    engine = engine or _load_engine()
    plain, shaped = [], []
    for side in SIDES:
        base = bank_vectors(engine, side, FEATURE_VERSION_BASE)
        extra = bank_vectors(engine, side, FEATURE_VERSION_EXTENDED)
        a = leave_one_out_knn(base, k=k)
        b = leave_one_out_knn(extra, k=k)
        if a is not None:
            plain.append(a)
        if b is not None:
            shaped.append(b)
    return {
        "coords63": float(sum(plain) / len(plain)) if plain else None,
        "coords_plus_shape": float(sum(shaped) / len(shaped)) if shaped else None,
        "feature_default": FEATURE_VERSION_BASE,
    }


def _fit_sklearn(name, x, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    if name == "svm":
        try:
            model = LinearSVC(C=1.0, max_iter=4000, dual="auto")
        except TypeError:
            model = LinearSVC(C=1.0, max_iter=4000)
    else:
        model = RandomForestClassifier(n_estimators=40, max_depth=12, random_state=0, n_jobs=1)
    model.fit(x, y)
    return model


def leave_one_out_hand(rows, k=5, include_sklearn=True):
    """Honest LOO for one hand. rows: (vector, label)."""
    labels = [row[1] for row in rows]
    classes = sorted(set(labels))
    n = len(rows)
    summary = {
        "samples": n,
        "classes": len(classes),
        "class_ids": classes,
        "knn": _empty_model(samples=n, classes=len(classes)),
        "svm": _empty_model(samples=n, classes=len(classes)),
        "random_forest": _empty_model(samples=n, classes=len(classes)),
        "ensemble": _empty_model(samples=n, classes=len(classes)),
        "votes": [],
    }
    if n < 2:
        reason = "too few samples for leave-one-out"
        for key in ("knn", "svm", "random_forest", "ensemble"):
            summary[key] = _empty_model(reason, n, len(classes))
        return summary
    data = np.stack([row[0] for row in rows], axis=0)
    voter = EnsembleClassifier(mode="weighted", min_confidence=0.0, class_margin=0.0)

    knn_hits = knn_errors = knn_unknown = 0
    knn_predict_ms = 0.0
    knn_train_ms = 0.0
    for index in range(n):
        mask = np.ones(n, dtype=bool)
        mask[index] = False
        start = time.perf_counter()
        train_x, train_y = data[mask], [labels[i] for i, keep in enumerate(mask) if keep]
        knn_train_ms += (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        predicted = _knn_predict(train_x, train_y, data[index], k)
        knn_predict_ms += (time.perf_counter() - start) * 1000.0
        if predicted is None or predicted == UNKNOWN:
            knn_unknown += 1
        elif predicted == labels[index]:
            knn_hits += 1
        else:
            knn_errors += 1
    summary["knn"] = {
        "samples": n,
        "classes": len(classes),
        "accuracy": knn_hits / float(n),
        "errors": knn_errors,
        "unknown": knn_unknown,
        "ambiguous": 0,
        "train_ms": knn_train_ms,
        "predict_avg_ms": knn_predict_ms / float(n),
        "skipped": "",
        "disagreements": 0,
    }

    if not include_sklearn or not sklearn_available():
        reason = "sklearn unavailable" if include_sklearn else "sklearn disabled"
        for key in ("svm", "random_forest", "ensemble"):
            summary[key] = _empty_model(reason, n, len(classes))
        return summary
    if len(classes) < MIN_CLASSES:
        reason = f"{SKIP_INSUFFICIENT} — only {len(classes)} calibrated class" + ("" if len(classes) == 1 else "es")
        for key in ("svm", "random_forest", "ensemble"):
            summary[key] = _empty_model(reason, n, len(classes))
        return summary

    svm_hits = svm_errors = svm_unknown = 0
    rf_hits = rf_errors = rf_unknown = 0
    ens_hits = ens_errors = ens_unknown = ens_ambiguous = 0
    svm_train_ms = svm_predict_ms = 0.0
    rf_train_ms = rf_predict_ms = 0.0
    ens_predict_ms = 0.0
    disagreements = 0
    vote_rows = []
    for index in range(n):
        mask = np.ones(n, dtype=bool)
        mask[index] = False
        train_x = data[mask]
        train_y = np.array([labels[i] for i, keep in enumerate(mask) if keep])
        query = data[index:index + 1]
        truth = labels[index]
        if len(set(train_y.tolist())) < MIN_CLASSES:
            svm_unknown += 1
            rf_unknown += 1
            ens_unknown += 1
            continue
        start = time.perf_counter()
        try:
            svm = _fit_sklearn("svm", train_x, train_y)
        except Exception:
            svm = None
        svm_train_ms += (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        try:
            rf = _fit_sklearn("rf", train_x, train_y)
        except Exception:
            rf = None
        rf_train_ms += (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        svm_pred = UNKNOWN
        if svm is not None:
            try:
                svm_pred = str(svm.predict(query)[0])
            except Exception:
                svm_pred = UNKNOWN
        svm_predict_ms += (time.perf_counter() - start) * 1000.0
        if svm_pred == UNKNOWN:
            svm_unknown += 1
        elif svm_pred == truth:
            svm_hits += 1
        else:
            svm_errors += 1

        start = time.perf_counter()
        rf_pred = UNKNOWN
        if rf is not None:
            try:
                rf_pred = str(rf.predict(query)[0])
            except Exception:
                rf_pred = UNKNOWN
        rf_predict_ms += (time.perf_counter() - start) * 1000.0
        if rf_pred == UNKNOWN:
            rf_unknown += 1
        elif rf_pred == truth:
            rf_hits += 1
        else:
            rf_errors += 1

        knn_pred = _knn_predict(train_x, [str(item) for item in train_y], data[index], k) or UNKNOWN
        votes = {"knn": (knn_pred, 1.0)}
        if svm_pred != UNKNOWN:
            votes["svm"] = (svm_pred, 1.0)
        if rf_pred != UNKNOWN:
            votes["random_forest"] = (rf_pred, 1.0)
        start = time.perf_counter()
        chosen, _conf, info = voter.decide(votes)
        ens_predict_ms += (time.perf_counter() - start) * 1000.0
        agreement = float(info.get("agreement", 0.0) or 0.0)
        unique = {knn_pred, svm_pred, rf_pred}
        if len(unique - {UNKNOWN}) > 1:
            disagreements += 1
        if info.get("ambiguous") or chosen == UNKNOWN:
            ens_unknown += 1
            ens_ambiguous += 1 if info.get("ambiguous") else 0
        elif chosen == truth:
            ens_hits += 1
        else:
            ens_errors += 1
        if index < 16 or knn_pred != svm_pred or svm_pred != rf_pred:
            vote_rows.append({
                "index": index,
                "truth": truth,
                "knn": knn_pred,
                "svm": svm_pred,
                "random_forest": rf_pred,
                "ensemble": chosen,
                "agreement": agreement,
                "ambiguous": bool(info.get("ambiguous", False)),
                "active_models": list(info.get("active_models") or []),
            })

    def pack(hits, errors, unknown, train_ms, predict_ms, extra=None):
        payload = {
            "samples": n,
            "classes": len(classes),
            "accuracy": hits / float(n),
            "errors": errors,
            "unknown": unknown,
            "ambiguous": 0,
            "train_ms": train_ms,
            "predict_avg_ms": predict_ms / float(n),
            "skipped": "",
            "disagreements": 0,
        }
        if extra:
            payload.update(extra)
        return payload

    summary["svm"] = pack(svm_hits, svm_errors, svm_unknown, svm_train_ms, svm_predict_ms)
    summary["random_forest"] = pack(rf_hits, rf_errors, rf_unknown, rf_train_ms, rf_predict_ms)
    summary["ensemble"] = pack(
        ens_hits, ens_errors, ens_unknown, svm_train_ms + rf_train_ms, ens_predict_ms,
        extra={"ambiguous": ens_ambiguous, "disagreements": disagreements},
    )
    summary["votes"] = vote_rows
    return summary


def compare_sklearn(engine=None):
    """Optional SVM / RandomForest vs k-NN. None if sklearn is absent."""
    if not sklearn_available():
        return None
    engine = engine or _load_engine()
    rows = bank_vectors(engine, "LEFT") + bank_vectors(engine, "RIGHT")
    labels = [row[1] for row in rows]
    if len(set(labels)) < MIN_CLASSES:
        return {"skipped": SKIP_INSUFFICIENT}
    result = leave_one_out_hand(rows, k=5, include_sklearn=True)
    return {
        "knn5": result["knn"].get("accuracy"),
        "linear_svc": result["svm"].get("accuracy"),
        "random_forest": result["random_forest"].get("accuracy"),
        "ensemble": result["ensemble"].get("accuracy"),
        "skipped": result["svm"].get("skipped") or "",
    }


def run_ml_benchmark(path=None, k=5, feature_versions=None):
    """Full per-hand LOO report. Dataset is always gestures.json unless path is given.

    SVM/RF/Ensemble use FEATURE_VERSION_BASE (live 63-d). Extended 10 distances
    are compared with k-NN only; duplicating sklearn LOO on extra features is
    not worth the extra train time unless a later benchmark asks for it.
    """
    path = path or user_data_path(GESTURES_FILE)
    engine = _load_engine(path)
    report = {
        "dataset": os.path.basename(path) if path else GESTURES_FILE,
        "dataset_path": path,
        "sklearn": sklearn_available(),
        "feature_size": FEATURE_SIZE,
        "method": "leave-one-out",
        "note": "Au moins 2 gestes calibrés distincts sont nécessaires pour entraîner SVM/RF.",
        "hands": {},
        "features": {},
    }
    if engine is None:
        report["error"] = "gestures.json missing"
        return report
    for side in SIDES:
        report["hands"][side] = leave_one_out_hand(
            bank_vectors(engine, side, FEATURE_VERSION_BASE), k=k,
        )
    report["features"][FEATURE_VERSION_BASE] = report["hands"]
    report["shape_compare"] = compare_shape_features(engine, k=k)
    return report


def _pct(value):
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def format_ml_benchmark(report, vote_limit=8):
    lines = [
        "========================================",
        "HANDCONTROLLER ML BENCHMARK",
        "========================================",
        f"Dataset: {report.get('dataset', GESTURES_FILE)}",
        "Method: leave-one-out (train never contains the tested sample)",
        f"sklearn: {'yes' if report.get('sklearn') else 'no'}",
        "Hands: LEFT / RIGHT",
        "Models: KNN / SVM / Random Forest / Ensemble",
        report.get("note", ""),
        "----------------------------------------",
    ]
    if report.get("error"):
        lines.append(str(report["error"]))
        return "\n".join(lines)
    for side in SIDES:
        block = (report.get("hands") or {}).get(side) or {}
        lines.append(f"{side}")
        lines.append(f"Samples: {block.get('samples', 0)}")
        lines.append(f"Classes: {block.get('classes', 0)}  {', '.join(block.get('class_ids') or [])}")
        for name, title in (
            ("knn", "KNN"),
            ("svm", "SVM"),
            ("random_forest", "RANDOM FOREST"),
            ("ensemble", "ENSEMBLE"),
        ):
            stats = block.get(name) or {}
            lines.append("----------------------------------------")
            lines.append(title)
            if stats.get("skipped"):
                lines.append(f"{side}: {title} skipped — {stats['skipped']}")
                continue
            lines.append(f"Samples: {stats.get('samples')}")
            lines.append(f"Accuracy: {_pct(stats.get('accuracy'))}")
            lines.append(f"Errors: {stats.get('errors')}")
            lines.append(f"UNKNOWN: {stats.get('unknown')}")
            lines.append(f"Train time: {float(stats.get('train_ms') or 0.0):.1f} ms")
            lines.append(f"Prediction latency: {float(stats.get('predict_avg_ms') or 0.0):.3f} ms")
            if name == "ensemble":
                lines.append(f"Disagreements: {stats.get('disagreements')}")
                lines.append(f"Ambiguous: {stats.get('ambiguous')}")
        votes = block.get("votes") or []
        if votes:
            lines.append("Votes:")
            for row in votes[:vote_limit]:
                lines.append(f"Sample #{row['index']}")
                lines.append(f"  truth: {row['truth']}")
                lines.append(f"  KNN: {row['knn']}")
                lines.append(f"  SVM: {row['svm']}")
                lines.append(f"  Random Forest: {row['random_forest']}")
                lines.append(f"  Ensemble: {row['ensemble']}")
                lines.append(f"  Agreement: {100.0 * float(row.get('agreement') or 0.0):.1f}%")
                lines.append(f"  Ambiguous: {row.get('ambiguous')}")
        lines.append("========================================")
    shape = report.get("shape_compare") or {}
    lines.append("FEATURES")
    lines.append(f"Default live: {FEATURE_VERSION_BASE} ({FEATURE_SIZE} coords)")
    lines.append(f"Extended: {FEATURE_VERSION_EXTENDED} (63 coords + 10 distances)")
    lines.append(f"KNN coords63: {_pct(shape.get('coords63'))}")
    lines.append(f"KNN coords+shape: {_pct(shape.get('coords_plus_shape'))}")
    lines.append("If the extra distances do not improve accuracy, keep the 63-d vector live.")
    lines.append("Next recognition gains should come from sample variety, not more features.")
    lines.append("========================================")
    return "\n".join(lines)
