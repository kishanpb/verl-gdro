from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from verl.protocol import DataProto

# Lightweight module-level state to accumulate per-category difficulty across steps
__knapsack_gdro_state = {
    "cum_difficulty": {},  # dict[str, float] - legacy cumulative
    "ema_difficulty": {},  # dict[str, float] - EMA difficulty (new)
    "counts": {},          # dict[str, int]
}

# Online pass@k tracking state (per-UID)
__passk_state = {
    "uid_acc": {},   # dict[str, float] - EMA pass probability per UID
    "uid_cnt": {},   # dict[str, int] - update counts
}

def update_online_passk(uids: list[str], any_correct: list[float], ema_beta: float = 0.15) -> None:
    """Update online pass@k EMA per-UID from aggregated any-of-k correctness.

    Args:
        uids: list of stable prompt identifiers
        any_correct: list of 0/1 (float/bool) values for pass@k at this step
        ema_beta: smoothing factor for EMA updates
    """
    if not uids or not any_correct:
        return
    for uid, val in zip(uids, any_correct, strict=True):
        try:
            v = float(val)
        except Exception:
            v = 0.0
        old = __passk_state["uid_acc"].get(uid, v)
        new = (1.0 - ema_beta) * old + ema_beta * v
        __passk_state["uid_acc"][uid] = new
        __passk_state["uid_cnt"][uid] = __passk_state["uid_cnt"].get(uid, 0) + 1


def _safe_entropy(probs: np.ndarray) -> float:
    p = probs.astype(float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _gini(x: np.ndarray) -> float:
    v = x.astype(float)
    n = v.size
    if n == 0:
        return 0.0
    mean = float(v.mean())
    if mean == 0:
        return 0.0
    diffsum = np.abs(v[:, None] - v[None, :]).sum()
    return float(diffsum / (2.0 * n * n * mean))


def _build_indices_from_budgets(budgets: np.ndarray) -> List[int]:
    indices: List[int] = []
    for task_id, task_budget in enumerate(budgets.tolist()):
        if task_budget > 0:
            indices.extend([task_id] * int(task_budget))
    return indices


def budget_allocation_vanilla(batch: DataProto, total_budget: int) -> Tuple[DataProto, np.ndarray]:
    n = max(1, int(len(batch)))
    base = total_budget // n
    rem = total_budget - base * n
    budgets = np.full(n, base, dtype=int)
    if rem > 0:
        budgets[:rem] += 1
    indices = _build_indices_from_budgets(budgets)
    selected = batch.select_idxs(indices)
    # Log minimal allocation stats
    total = float(budgets.sum()) if budgets.sum() > 0 else 1.0
    probs = budgets.astype(float) / total
    selected.meta_info["knapsack_metrics"] = {
        "method": "vanilla",
        "alloc_min": int(budgets.min()) if budgets.size else 0,
        "alloc_max": int(budgets.max()) if budgets.size else 0,
        "alloc_mean": float(budgets.mean()) if budgets.size else 0.0,
        "alloc_nonzero": int((budgets > 0).sum()),
        "alloc_entropy": _safe_entropy(probs),
        "alloc_gini": _gini(budgets),
    }
    return selected, budgets


def _proportional_round(scores: np.ndarray, total: int) -> np.ndarray:
    scores = np.maximum(0.0, scores.astype(float))
    if scores.sum() <= 0:
        # fall back to equal
        n = scores.shape[0]
        base = total // n
        rem = total - base * n
        out = np.full(n, base, dtype=int)
        if rem > 0:
            out[:rem] += 1
        return out
    probs = scores / scores.sum()
    raw = probs * float(total)
    floors = np.floor(raw).astype(int)
    remain = total - int(floors.sum())
    if remain > 0:
        fracs = raw - floors
        order = np.argsort(-fracs)
        floors[order[:remain]] += 1
    return floors.astype(int)


def _to_sequence(field: Any) -> Optional[List[Any]]:
    if field is None:
        return None
    if isinstance(field, np.ndarray):
        try:
            return field.tolist()
        except Exception:
            return list(field)
    if isinstance(field, (list, tuple)):
        return list(field)
    return None


def _get_seq_item(seq: Optional[List[Any]], idx: int) -> Any:
    if seq is None:
        return None
    if idx < 0 or idx >= len(seq):
        return None
    try:
        return seq[idx]
    except Exception:
        return None


def _normalize_uid_component(val: Any) -> str:
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="ignore")
        except Exception:
            return val.hex()
    return str(val)


def _dataset_style_uid(data_sources: Optional[List[Any]], extra: Any, idx: int) -> Optional[str]:
    ds = None
    if data_sources is not None and idx < len(data_sources):
        ds = data_sources[idx]
    if isinstance(extra, dict):
        ds = extra.get("data_source", ds)
    if ds is None:
        return None
    split = ""
    idx_val = idx
    if isinstance(extra, dict):
        split = extra.get("split", "")
        idx_val = extra.get("index", idx_val)
    try:
        return f"{_normalize_uid_component(ds)}:{_normalize_uid_component(split)}:{_normalize_uid_component(idx_val)}"
    except Exception:
        return None


def _lookup_passk_probabilities(batch: DataProto, n: int) -> Tuple[np.ndarray, Dict[str, int]]:
    pass_probs = np.zeros(n, dtype=float)
    uid_seq = _to_sequence(batch.non_tensor_batch.get("uid"))
    data_sources = _to_sequence(batch.non_tensor_batch.get("data_source"))
    extra_infos = _to_sequence(batch.non_tensor_batch.get("extra_info"))

    # Build quick lookup tables so idx-style identifiers can map to dataset-style pass@k keys
    suffix_map: Dict[str, List[str]] = defaultdict(list)
    try:
        for key in __passk_state["uid_acc"].keys():
            try:
                parts = str(key).split(":")
            except Exception:
                continue
            if not parts:
                continue
            suffix_map[parts[-1]].append(key)
            if len(parts) >= 2:
                suffix_map[":".join(parts[-2:])].append(key)
    except Exception:
        suffix_map.clear()

    direct_hits = 0
    alias_hits = 0

    for i in range(n):
        keys_to_try: List[Tuple[str, Optional[str]]] = []
        primary = _get_seq_item(uid_seq, i)
        if primary is not None:
            try:
                keys_to_try.append(("direct", _normalize_uid_component(primary)))
            except Exception:
                keys_to_try.append(("direct", primary))

        extra = _get_seq_item(extra_infos, i)
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = None
        if isinstance(extra, dict):
            extra_uid = extra.get("uid")
            if extra_uid is not None:
                try:
                    keys_to_try.append(("alias", _normalize_uid_component(extra_uid)))
                except Exception:
                    keys_to_try.append(("alias", extra_uid))

        dataset_uid = _dataset_style_uid(data_sources, extra, i)
        if dataset_uid is not None:
            keys_to_try.append(("alias", dataset_uid))

        if isinstance(primary, str) and suffix_map:
            primary_parts = primary.split(":")
            if primary_parts:
                last = primary_parts[-1]
                for candidate in suffix_map.get(last, []):
                    keys_to_try.append(("alias", candidate))
                if len(primary_parts) >= 2:
                    last_two = ":".join(primary_parts[-2:])
                    for candidate in suffix_map.get(last_two, []):
                        keys_to_try.append(("alias", candidate))

        found = False
        seen_keys = set()
        for label, key in keys_to_try:
            if key is None:
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                val = __passk_state["uid_acc"].get(key)
            except Exception:
                val = None
            if val is not None:
                pass_probs[i] = float(val)
                if label == "direct":
                    direct_hits += 1
                else:
                    alias_hits += 1
                found = True
                break
        if not found:
            pass_probs[i] = 0.0

    stats = {
        "direct_hits": int(direct_hits),
        "alias_hits": int(alias_hits),
        "misses": int(max(0, n - direct_hits - alias_hits)),
    }
    return pass_probs, stats


def _passk_difficulty_from_batch(batch: DataProto, n: int) -> Tuple[np.ndarray, str, Dict[str, int]]:
    pass_probs, stats = _lookup_passk_probabilities(batch, n)
    source = "passk" if (stats["direct_hits"] + stats["alias_hits"]) > 0 else "uniform"
    diffs = np.clip(1.0 - pass_probs, 0.0, 1.0)
    return diffs, source, stats


def budget_allocation_knapsack(
    batch: DataProto,
    total_budget: int,
    *,
    score_key: str = "status",
    scores: Optional[np.ndarray] = None,
    N_low: int = 2,
    N_up: int = 128,
) -> Tuple[DataProto, np.ndarray]:
    n = max(1, int(len(batch)))
    score_source = "score_key"
    score_stats: Dict[str, int] = {"direct_hits": 0, "alias_hits": 0, "misses": n}
    if scores is None:
        arr = None
        if isinstance(batch.non_tensor_batch.get(score_key), np.ndarray):
            arr = batch.non_tensor_batch.get(score_key)
        elif batch.batch is not None and score_key in batch.batch.keys():
            try:
                arr = batch.batch[score_key].detach().cpu().numpy()
            except Exception:
                arr = None
        valid_scores = None
        if arr is not None:
            candidate = np.array(arr, dtype=float)
            if candidate.ndim > 1:
                candidate = candidate.reshape(-1)
            if candidate.shape[0] == n and np.isfinite(candidate).all() and float(candidate.std()) > 1e-6:
                valid_scores = candidate
        if valid_scores is not None:
            scores = valid_scores
        else:
            scores, score_source, score_stats = _passk_difficulty_from_batch(batch, n)

    # Initial proportional allocation by difficulty/need
    budgets = _proportional_round(scores, int(total_budget))

    # Enforce Eq.(5) integer bounds: N_low <= N_i <= N_up
    # Guard when total budget cannot satisfy lower bounds
    n = scores.shape[0]
    hard_low = max(0, int(N_low))
    hard_up = max(hard_low, int(N_up))
    if hard_low * n > int(total_budget):
        # Relax lower bound uniformly so feasibility holds
        hard_low = int(total_budget) // max(1, n)
    budgets = budgets.astype(int)
    budgets = np.clip(budgets, hard_low, hard_up)

    # Fix drift created by clamping while respecting bounds
    drift = int(total_budget) - int(budgets.sum())
    if drift != 0:
        order_hi = np.argsort(-scores)  # prefer higher-need when adding
        order_lo = np.argsort(scores)   # prefer lower-need when removing
        i_hi, i_lo = 0, 0
        while drift != 0 and (i_hi < n or i_lo < n):
            if drift > 0:
                # add one to highest-need item that is < N_up
                j = order_hi[i_hi % n]
                if budgets[j] < hard_up:
                    budgets[j] += 1
                    drift -= 1
                i_hi += 1
            else:
                # remove one from lowest-need item that is > N_low
                j = order_lo[i_lo % n]
                if budgets[j] > hard_low:
                    budgets[j] -= 1
                    drift += 1
                i_lo += 1
    indices = _build_indices_from_budgets(budgets)
    selected = batch.select_idxs(indices)
    total = float(budgets.sum()) if budgets.sum() > 0 else 1.0
    probs = budgets.astype(float) / total
    corr = 0.0
    if budgets.size > 1 and scores.size > 1:
        try:
            c = float(np.corrcoef(scores, budgets)[0, 1])
            if not np.isnan(c):
                corr = c
        except Exception:
            corr = 0.0
    selected.meta_info["knapsack_metrics"] = {
        "method": "knapsack",
        "alloc_min": int(budgets.min()) if budgets.size else 0,
        "alloc_max": int(budgets.max()) if budgets.size else 0,
        "alloc_mean": float(budgets.mean()) if budgets.size else 0.0,
        "alloc_nonzero": int((budgets > 0).sum()),
        "alloc_entropy": _safe_entropy(probs),
        "alloc_gini": _gini(budgets),
        "corr_score_budget": corr,
    }
    return selected, budgets


def budget_allocation_knapsack_group_dro(
    batch: DataProto,
    total_budget: int,
    *,
    score_key: str = "status",
    category_key: Optional[str] = None,
    eta_q: float = 0.10,
    gamma: float = 0.10,
    ema_alpha: float = 0.15,
) -> Tuple[DataProto, np.ndarray]:
    # Derive categories
    n = int(len(batch))
    if category_key is None:
        category_key = score_key

    categories = None
    
    # Handle composite categories (e.g., "level_x_type")
    if "_x_" in category_key:
        # Split composite key (e.g., "level_x_type" -> ["level", "type"])
        cat_keys = [k.strip() for k in category_key.split("_x_")]
        
        # Get each component
        cat_components = []
        for key in cat_keys:
            comp_src = batch.non_tensor_batch.get(key)
            if isinstance(comp_src, np.ndarray) and comp_src.shape[0] == n and comp_src.dtype == object:
                cat_components.append(comp_src.astype(str))
            elif batch.batch is not None and key in batch.batch.keys():
                try:
                    arr = batch.batch[key].detach().cpu().numpy()
                    if arr.shape[0] == n:
                        cat_components.append(arr.astype(str))
                    else:
                        cat_components.append(None)
                except Exception:
                    cat_components.append(None)
            else:
                cat_components.append(None)
        
        # Combine components if all are available
        if all(comp is not None for comp in cat_components):
            categories = np.array([f"{comp[0]} x {comp[1]}" for comp in zip(*cat_components)])
        else:
            categories = None
    
    # Single category key (original logic)
    if categories is None:
        cat_src = batch.non_tensor_batch.get(category_key)
        if isinstance(cat_src, np.ndarray) and cat_src.shape[0] == n and cat_src.dtype == object:
            categories = cat_src.astype(str)
        # If not in non-tensor, try tensor -> treat ints as category ids
        elif batch.batch is not None and category_key in batch.batch.keys():
            try:
                arr = batch.batch[category_key].detach().cpu().numpy()
                if arr.shape[0] == n:
                    categories = arr.astype(str)
            except Exception:
                categories = None

    # If no categories available, fall back to plain knapsack
    if categories is None:
        return budget_allocation_knapsack(batch, total_budget, score_key=score_key)

    uniq = np.unique(categories)
    if uniq.size == 0:
        return budget_allocation_knapsack(batch, total_budget, score_key=score_key)

    # Read scores (difficulty proxies)
    score_source = "score_key"
    score_stats: Dict[str, int] = {"direct_hits": 0, "alias_hits": 0, "misses": n}
    arr = None
    if isinstance(batch.non_tensor_batch.get(score_key), np.ndarray):
        arr = batch.non_tensor_batch.get(score_key)
    elif batch.batch is not None and score_key in batch.batch.keys():
        try:
            arr = batch.batch[score_key].detach().cpu().numpy()
        except Exception:
            arr = None
    valid_scores = None
    if arr is not None:
        candidate = np.array(arr, dtype=float)
        if candidate.ndim > 1:
            candidate = candidate.reshape(-1)
        if candidate.shape[0] == n and np.isfinite(candidate).all() and float(candidate.std()) > 1e-6:
            valid_scores = candidate
    if valid_scores is not None:
        scores = valid_scores
    else:
        scores, score_source, score_stats = _passk_difficulty_from_batch(batch, n)

    # Update EMA difficulty per category (use mean score per category as step signal)
    cat_to_idx = {c: np.where(categories == c)[0] for c in uniq}
    for c, idxs in cat_to_idx.items():
        if idxs.size == 0:
            continue
        step_signal = float(np.mean(scores[idxs]))
        
        # EMA update: ema = (1 - alpha) * old_ema + alpha * step_signal
        old_ema = __knapsack_gdro_state["ema_difficulty"].get(c, step_signal)
        new_ema = (1.0 - ema_alpha) * old_ema + ema_alpha * step_signal
        __knapsack_gdro_state["ema_difficulty"][c] = new_ema
        
        # Keep counts for debugging
        __knapsack_gdro_state["counts"][c] = __knapsack_gdro_state["counts"].get(c, 0) + 1
    # EXP3P weights over categories using EMA difficulty (bounded and stable)
    expw = {}
    for c in uniq:
        ema_difficulty = float(__knapsack_gdro_state["ema_difficulty"].get(c, 0.0))
        expw[c] = np.exp(eta_q * ema_difficulty)
    Z = float(sum(expw.values())) or 1.0
    w_class = {c: (1.0 - gamma) * (expw[c] / Z) + gamma / float(len(uniq)) for c in uniq}

    # Allocate class-level budgets
    class_shares = np.array([w_class[c] for c in uniq], dtype=float)
    class_budgets = _proportional_round(class_shares, int(total_budget))

    # Within each class, allocate proportionally to per-sample scores (need factor)
    budgets = np.zeros(n, dtype=int)
    for c, class_budget in zip(uniq, class_budgets.tolist()):
        idxs = cat_to_idx[c]
        if idxs.size == 0 or class_budget <= 0:
            continue
        need = scores[idxs]
        # Avoid all-zero division
        if float(need.sum()) <= 0:
            per = np.ones_like(need, dtype=float)
        else:
            per = need.astype(float)
        per_alloc = _proportional_round(per, int(class_budget))
        budgets[idxs] = budgets[idxs] + per_alloc

    # Ensure total sums to total_budget (fix rounding drift)
    drift = int(total_budget) - int(budgets.sum())
    if drift != 0:
        # Distribute drift to highest-need samples globally
        order = np.argsort(-scores)
        i = 0
        while drift != 0 and i < len(order):
            j = order[i]
            if drift > 0:
                budgets[j] += 1
                drift -= 1
            else:
                if budgets[j] > 0:
                    budgets[j] -= 1
                    drift += 1
            i += 1

    indices = _build_indices_from_budgets(budgets)
    selected = batch.select_idxs(indices)
    total = float(budgets.sum()) if budgets.sum() > 0 else 1.0
    probs = budgets.astype(float) / total
    class_probs = np.array([w_class[c] for c in uniq], dtype=float)
    
    # Compute score-budget correlation
    corr_score_budget = 0.0
    if budgets.size > 1 and scores.size > 1:
        try:
            corr_score_budget = float(np.corrcoef(scores, budgets)[0, 1])
            if np.isnan(corr_score_budget):
                corr_score_budget = 0.0
        except Exception:
            corr_score_budget = 0.0
    
    knapsack_metrics = {
        "method": "knapsack_group_dro",
        "alloc_min": int(budgets.min()) if budgets.size else 0,
        "alloc_max": int(budgets.max()) if budgets.size else 0,
        "alloc_mean": float(budgets.mean()) if budgets.size else 0.0,
        "alloc_nonzero": int((budgets > 0).sum()),
        "alloc_entropy": _safe_entropy(probs),
        "alloc_gini": _gini(budgets),
        "corr_score_budget": corr_score_budget,
        "num_classes": int(len(uniq)),
        "class_weight_entropy": _safe_entropy(class_probs / float(class_probs.sum() or 1.0)),
        "top_class_share": float(class_probs.max()) if class_probs.size else 0.0,
    }
    selected.meta_info["knapsack_metrics"] = knapsack_metrics
    return selected, budgets


def _parse_edges(edges: list[float] | None) -> list[float]:
    if not edges:
        # Default bin edges similar to Prompt-GDRO setups
        return [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    return sorted([float(x) for x in edges if 0.0 <= float(x) <= 1.0])


def _assign_bins(passk: np.ndarray, edges: list[float]) -> tuple[np.ndarray, list[str]]:
    # B bins defined by edges e: [0, e1), [e1, e2), ..., [e_{m}, 1]
    labels = []
    prev = 0.0
    for e in edges:
        labels.append(f"[{prev:.2f},{e:.2f})")
        prev = e
    labels.append(f"[{prev:.2f},1.00]")
    bins = np.digitize(passk, edges, right=False)  # 0..len(edges)
    return bins.astype(int), labels


def budget_allocation_knapsack_gdro_passk(
    batch: DataProto,
    total_budget: int,
    *,
    passk_edges: Optional[list[float]] = None,
    eta_q: float = 0.10,
    gamma: float = 0.10,
    ema_alpha: float = 0.15,
    passk_focus_enable: bool = False,
    passk_focus_min: float = 0.05,
) -> tuple[DataProto, np.ndarray]:
    """Knapsack + Rollout-GDRO using online pass@k bins as classes.

    - Per-UID pass probability p_i from EMA in __passk_state
    - Difficulty d_i = 1 - p_i
    - Bin difficulty EMA: D_b ← (1-α)D_b + α*(1 - mean p_i in bin)
    - Class share via EXP3P on D_b; within-bin proportional to d_i
    """
    n = int(len(batch))
    if n <= 0:
        return budget_allocation_vanilla(batch, total_budget)

    # Gather per-sample pass@k estimates via state lookup (default 0.0 if unseen)
    p, passk_stats = _lookup_passk_probabilities(batch, n)
    d = np.clip(1.0 - p, 0.0, 1.0)

    # Build bins
    edges = _parse_edges(passk_edges)
    bin_idx, labels = _assign_bins(p, edges)
    uniq_bins = np.unique(bin_idx)

    # Maintain per-bin EMA difficulty state locally (reuse knapsack gdro state with distinct keys)
    bin_key = "passk_bin_ema"
    if bin_key not in __knapsack_gdro_state:
        __knapsack_gdro_state[bin_key] = {}

    # Update bin EMA difficulty: 1 - mean pass@k
    for b in uniq_bins:
        idxs = np.where(bin_idx == b)[0]
        if idxs.size == 0:
            continue
        mean_p = float(np.mean(p[idxs]))
        step_signal = 1.0 - mean_p
        label = labels[int(b)] if 0 <= int(b) < len(labels) else str(b)
        old = __knapsack_gdro_state[bin_key].get(label, step_signal)
        new = (1.0 - ema_alpha) * old + ema_alpha * step_signal
        __knapsack_gdro_state[bin_key][label] = new

    # Compute EXP3P class weights over bins
    bin_labels = [labels[int(b)] if 0 <= int(b) < len(labels) else str(b) for b in uniq_bins]
    expw = {}
    for lab in bin_labels:
        ema_dif = float(__knapsack_gdro_state[bin_key].get(lab, 0.0))
        expw[lab] = np.exp(eta_q * ema_dif)
    Z = float(sum(expw.values())) or 1.0
    w_bin = {lab: (1.0 - gamma) * (expw[lab] / Z) + gamma / float(max(1, len(bin_labels))) for lab in bin_labels}

    # Optional focus schedule: downweight extremes (< first edge, > last edge)
    if passk_focus_enable and len(labels) >= 2:
        first_lab = labels[0]
        last_lab = labels[-1]
        changed = False
        if first_lab in w_bin:
            w_bin[first_lab] = max(0.0, float(passk_focus_min)) * w_bin[first_lab]
            changed = True
        if last_lab in w_bin:
            w_bin[last_lab] = max(0.0, float(passk_focus_min)) * w_bin[last_lab]
            changed = True
        if changed:
            s = float(sum(w_bin.values())) or 1.0
            for k in w_bin:
                w_bin[k] = w_bin[k] / s

    # Allocate budgets per bin
    shares = np.array([w_bin[labels[int(b)]] for b in uniq_bins], dtype=float)
    bin_budgets = _proportional_round(shares, int(total_budget))

    budgets = np.zeros(n, dtype=int)
    for b, B_b in zip(uniq_bins, bin_budgets.tolist()):
        idxs = np.where(bin_idx == b)[0]
        if idxs.size == 0 or B_b <= 0:
            continue
        need = d[idxs]
        per = need + 1e-6  # avoid all-zero
        alloc = _proportional_round(per, int(B_b))
        budgets[idxs] = budgets[idxs] + alloc

    # Fix drift
    drift = int(total_budget) - int(budgets.sum())
    if drift != 0:
        order = np.argsort(-d)  # prioritize harder
        i = 0
        while drift != 0 and i < len(order):
            j = order[i]
            if drift > 0:
                budgets[j] += 1
                drift -= 1
            else:
                if budgets[j] > 0:
                    budgets[j] -= 1
                    drift += 1
            i += 1

    indices = _build_indices_from_budgets(budgets)
    selected = batch.select_idxs(indices)

    total = float(budgets.sum()) if budgets.sum() > 0 else 1.0
    probs = budgets.astype(float) / total
    class_probs = np.array([w_bin[lab] for lab in bin_labels], dtype=float)

    # Metrics
    selected.meta_info["knapsack_metrics"] = {
        "method": "knapsack_gdro_passk",
        "alloc_min": int(budgets.min()) if budgets.size else 0,
        "alloc_max": int(budgets.max()) if budgets.size else 0,
        "alloc_mean": float(budgets.mean()) if budgets.size else 0.0,
        "alloc_nonzero": int((budgets > 0).sum()),
        "alloc_entropy": _safe_entropy(probs),
        "alloc_gini": _gini(budgets),
        "num_classes": int(len(bin_labels)),
        "class_weight_entropy": _safe_entropy(class_probs / float(class_probs.sum() or 1.0)),
        "top_class_share": float(class_probs.max()) if class_probs.size else 0.0,
    }
    return selected, budgets


