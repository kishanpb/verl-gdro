from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import math


@dataclass
class Exp3pConfig:
    eta_q: float = 0.1
    gamma: float = 0.15
    max_class_weight: float = 10.0
    prompt_classifier: str = "gsm8k"  # "gsm8k", "math", "custom"
    debias_scores: bool = False  # if True, use (cumulative / count) per class
    # If True, use EMA of per-step class mean loss as score (takes precedence over debias_scores)
    debias_scores_ema: bool = False
    ema_beta: float = 0.1
    # Optional: z-score the class score (center/scale) before exponentiation
    use_zscore: bool = False
    z_std_floor: float = 1e-3
    z_cap: float = 3.0
    # Online pass@k classifier knobs
    passk_num_bins: int = 10
    passk_history_len: int = 0  # 0 = unlimited (cumulative)
    passk_hysteresis: float = 0.0  # fraction in [0, 0.5). Adds inertia to bin switches
    # Optional custom bin edges in (0,1), CSV like "0.25,0.75" → 3 bins
    passk_edges: str = ""
    # If True, exclude exact 0% and 100% success prompts from weighting
    passk_exclude_extremes: bool = False
    # Loss update normalization
    loss_norm_by_class: bool = False  # if True, normalize per-step class loss by its batch share
    # Focus mask over accuracy bins (apply after EXP3P weights)
    passk_focus_enable: bool = False
    # CSV mapping like "0:0.1,1:0.1,8:0.7,9:0.7" (unspecified bins use 1.0)
    passk_focus_map: str = ""
    passk_focus_warmup_steps: int = 0
    passk_focus_ramp_steps: int = 0


class _BasePromptClassifier:
    def classify(self, prompt_text: str, metadata: Optional[dict] = None) -> str:
        raise NotImplementedError


class GSM8KClassifier(_BasePromptClassifier):
    def classify(self, prompt_text: str, metadata: Optional[dict] = None) -> str:
        # simple heuristic: difficulty by length, type by keywords
        txt = (prompt_text or "").lower()
        wc = len(txt.split())
        difficulty = "easy" if wc < 30 else ("medium" if wc < 60 else "hard")
        if any(w in txt for w in ["area", "perimeter", "rectangle", "triangle", "circle"]):
            ptype = "geometry"
        elif any(w in txt for w in ["percent", "%", "discount", "ratio", "percentage"]):
            ptype = "percentage"
        else:
            ptype = "arithmetic"
        return f"{difficulty}_{ptype}"


class MATHClassifier(_BasePromptClassifier):
    def classify(self, prompt_text: str, metadata: Optional[dict] = None) -> str:
        # Prefer explicit metadata columns from lighteval: 'type' and 'level'
        # examples/data_preprocess/math_dataset.py currently drops them; we infer from extra_info if present
        if isinstance(metadata, dict):
            # Try to pass-through if upstream added these fields to metadata
            lvl_str = metadata.get("level") or metadata.get("Level")
            typ_str = metadata.get("type") or metadata.get("Type")
            if isinstance(lvl_str, str) and lvl_str.lower().startswith("level") and isinstance(typ_str, str):
                # Normalize spaces and ampersands
                subj = (
                    typ_str.replace(" ", "").replace("&", "And").replace("-", "").title()
                )
                # Extract digits from Level X
                import re as _re
                m = _re.search(r"(\d+)", lvl_str)
                lvl = int(m.group(1)) if m else 1
                return f"{subj}_Level{lvl}"
            # Back-compat: subject/level if present
            subj = metadata.get("subject")
            lvl = metadata.get("level")
            if subj is not None and lvl is not None:
                return f"{subj}_Level{lvl}"
        # Heuristic fallback
        subj = None
        lvl = None
        if isinstance(metadata, dict):
            ds = metadata.get("data_source", "")
            if isinstance(ds, (list, tuple)) and len(ds) > 0:
                ds = ds[0]
            text = (metadata.get("raw_prompt") or prompt_text or "")
            low = (str(ds) + "\n" + str(text)).lower()
            if any(k in low for k in ["algebra"]):
                subj = "Algebra"
            elif any(k in low for k in ["geometry", "triangle", "circle", "angle"]):
                subj = "Geometry"
            elif any(k in low for k in ["number theory", "prime", "divisible", "mod"]):
                subj = "NumberTheory"
            elif any(k in low for k in ["combinatorics", "permutation", "combination", "choose"]):
                subj = "Combinatorics"
            elif any(k in low for k in ["probability", "expected", "probability of"]):
                subj = "Probability"
            else:
                subj = "unknown"
            wc = len(((metadata.get("raw_prompt") or prompt_text or "").split()))
            lvl = 1 if wc < 80 else (2 if wc < 160 else 3)
            return f"{subj}_Level{lvl}"
        # No metadata: heuristic by prompt text only
        low = (prompt_text or "").lower()
        if any(k in low for k in ["algebra"]):
            subj = "Algebra"
        elif any(k in low for k in ["geometry", "triangle", "circle", "angle"]):
            subj = "Geometry"
        elif any(k in low for k in ["number theory", "prime", "divisible", "mod"]):
            subj = "NumberTheory"
        elif any(k in low for k in ["combinatorics", "permutation", "combination", "choose"]):
            subj = "Combinatorics"
        elif any(k in low for k in ["probability", "expected", "probability of"]):
            subj = "Probability"
        else:
            subj = "unknown"
        wc = len((prompt_text or "").split())
        lvl = 1 if wc < 80 else (2 if wc < 160 else 3)
        return f"{subj}_Level{lvl}"


class MATHClassifier_Reduced1(_BasePromptClassifier):
    """MATH classifier with coarse taxonomy (Setup 6):
    - Merge subjects Precalculus and Prealgebra -> Precalc
    - Merge levels 1 and 2 -> 2 (keep 3/4/5 as-is)
    """
    def _coarsen(self, subject: Optional[str], level: Optional[int | str]) -> str:
        subj = str(subject or "unknown")
        subj_low = subj.lower()
        if subj_low in {"precalculus", "pre-algebra", "prealgebra"}:
            subj = "Precalc"
        # level normalization
        try:
            lvl_num = int(level) if level is not None else None
        except Exception:
            lvl_num = None
        if isinstance(lvl_num, int):
            if lvl_num <= 2:
                lvl_num = 2
        return f"{subj}_Level{lvl_num if lvl_num is not None else 'NA'}"

    def classify(self, prompt_text: str, metadata: Optional[dict] = None) -> str:
        # Prefer explicit metadata columns
        if isinstance(metadata, dict):
            lvl_str = metadata.get("level") or metadata.get("Level")
            typ_str = metadata.get("type") or metadata.get("Type")
            if isinstance(lvl_str, str) and isinstance(typ_str, str):
                # Normalize subject formatting
                subj = (
                    typ_str.replace(" ", "").replace("&", "And").replace("-", "").title()
                )
                # Extract digits from Level X
                import re as _re
                m = _re.search(r"(\d+)", lvl_str)
                lvl = int(m.group(1)) if m else 1
                return self._coarsen(subj, lvl)
            # Back-compat: subject/level if present
            subj = metadata.get("subject")
            lvl = metadata.get("level")
            if subj is not None and lvl is not None:
                return self._coarsen(subj, lvl)
        # Heuristic fallback (same as base but coarsened)
        subj = None
        lvl = None
        if isinstance(metadata, dict):
            ds = metadata.get("data_source", "")
            if isinstance(ds, (list, tuple)) and len(ds) > 0:
                ds = ds[0]
            text = (metadata.get("raw_prompt") or prompt_text or "")
            low = (str(ds) + "\n" + str(text)).lower()
            if any(k in low for k in ["algebra"]):
                subj = "Algebra"
            elif any(k in low for k in ["geometry", "triangle", "circle", "angle"]):
                subj = "Geometry"
            elif any(k in low for k in ["number theory", "prime", "divisible", "mod"]):
                subj = "NumberTheory"
            elif any(k in low for k in ["combinatorics", "permutation", "combination", "choose"]):
                subj = "Combinatorics"
            elif any(k in low for k in ["probability", "expected", "probability of"]):
                subj = "Probability"
            else:
                subj = "unknown"
            wc = len(((metadata.get("raw_prompt") or prompt_text or "").split()))
            lvl = 1 if wc < 80 else (2 if wc < 160 else 3)
            return self._coarsen(subj, lvl)
        # No metadata: heuristic by prompt text only
        low = (prompt_text or "").lower()
        if any(k in low for k in ["algebra"]):
            subj = "Algebra"
        elif any(k in low for k in ["geometry", "triangle", "circle", "angle"]):
            subj = "Geometry"
        elif any(k in low for k in ["number theory", "prime", "divisible", "mod"]):
            subj = "NumberTheory"
        elif any(k in low for k in ["combinatorics", "permutation", "combination", "choose"]):
            subj = "Combinatorics"
        elif any(k in low for k in ["probability", "expected", "probability of"]):
            subj = "Probability"
        else:
            subj = "unknown"
        wc = len((prompt_text or "").split())
        lvl = 1 if wc < 80 else (2 if wc < 160 else 3)
        return self._coarsen(subj, lvl)


class MATHClassifier_QwenClassified(_BasePromptClassifier):
    """Length-bin (or external model classified) taxonomy.
    Expects the per-sample metadata to contain a precomputed key 'lenbin'
    such as 'lenbin_0'..'lenbin_9'. Falls back to 'lenbin_unk'.
    """
    def classify(self, prompt_text: str, metadata: Optional[dict] = None) -> str:
        if isinstance(metadata, dict):
            val = metadata.get("lenbin") or metadata.get("length_bin") or metadata.get("qwen_lenbin")
            if isinstance(val, str) and len(val) > 0:
                return val
        return "lenbin_unk"


class PassKOnlineClassifier(_BasePromptClassifier):
    """Dynamic classifier that buckets prompts by online pass@k accuracy.

    Expects the per-sample metadata to contain a stable 'uid' for the prompt.
    Uses the parent ClassDroExp3p instance's running uid->accuracy mapping.
    """

    def __init__(
        self,
        parent: "ClassDroExp3p",
        num_bins: int = 10,
        history_len: int = 0,
        hysteresis: float = 0.0,
        edges: Optional[List[float]] = None,
        exclude_extremes: bool = False,
    ):
        self.parent = parent
        self.exclude_extremes = bool(exclude_extremes)
        self.edges = None
        if isinstance(edges, list) and len(edges) > 0:
            try:
                es = [float(x) for x in edges if 0.0 < float(x) < 1.0]
                es = sorted(list(set(es)))
                if len(es) >= 1:
                    self.edges = es
            except Exception:
                self.edges = None
        self.num_bins = max(2, int(num_bins if self.edges is None else (len(self.edges) + 1)))
        self.history_len = max(0, int(history_len))
        self.hysteresis = max(0.0, min(0.49, float(hysteresis)))
        # Optional sliding window of recent correctness per uid
        from collections import deque
        self._uid_recent: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_len if self.history_len > 0 else 0))
        self._last_bin: Dict[str, int] = {}

    def classify(self, prompt_text: str, metadata: Optional[dict] = None) -> str:
        uid = None
        if isinstance(metadata, dict):
            v = metadata.get("uid")
            if v is not None:
                uid = str(v)
        if not uid:
            return "accbin_unk"
        # Compute running accuracy; if history_len>0 and we have recent window, use it
        acc = None
        if self.history_len > 0:
            win = self._uid_recent.get(uid)
            if win and len(win) > 0:
                acc = float(sum(win)) / float(len(win))
        if acc is None:
            acc = self.parent._get_uid_accuracy(uid)
        # Optionally exclude exact 0% or 100% success prompts
        try:
            if self.exclude_extremes and (float(acc) <= 0.0 or float(acc) >= 1.0):
                return "accbin_excl"
        except Exception:
            pass
        # Map accuracy in [0,1] to discrete bins [0..num_bins-1]
        try:
            aval = float(acc)
            if self.edges is None:
                raw_bin = int(max(0, min(self.num_bins - 1, math.floor(aval * self.num_bins))))
            else:
                # edges define upper boundaries of lower bins
                raw_bin = 0
                for i, b in enumerate(self.edges):
                    if aval < b:
                        raw_bin = i
                        break
                else:
                    raw_bin = len(self.edges)
                raw_bin = int(max(0, min(self.num_bins - 1, raw_bin)))
        except Exception:
            raw_bin = 0
        # Apply hysteresis: resist switching across a boundary unless acc moves past a margin
        if self.hysteresis > 0.0:
            prev = self._last_bin.get(uid, raw_bin)
            if raw_bin != prev:
                # Boundary between k and k+1
                if self.edges is None:
                    boundary = (min(prev, raw_bin)) / self.num_bins if raw_bin < prev else (max(prev, raw_bin)) / self.num_bins
                else:
                    # For custom edges, boundary is edges[min(prev, raw_bin)]
                    idx = min(prev, raw_bin)
                    if idx < len(self.edges):
                        boundary = float(self.edges[idx])
                    else:
                        boundary = float(self.edges[-1]) if len(self.edges) > 0 else (prev + 1) / self.num_bins
                if raw_bin > prev:
                    if float(acc) < (boundary + self.hysteresis):
                        raw_bin = prev
                else:
                    if float(acc) > (boundary - self.hysteresis):
                        raw_bin = prev
            self._last_bin[uid] = raw_bin
        return f"accbin_{raw_bin}"

def build_classifier(name: str) -> _BasePromptClassifier:
    name = (name or "").lower()
    if name == "gsm8k":
        return GSM8KClassifier()
    if name == "math":
        return MATHClassifier()
    if name in {"math_reduced1", "math-reduced1", "math_reduced_1"}:
        return MATHClassifier_Reduced1()
    if name in {"math_qwenclassified", "math_len10", "math_qwencl"}:
        return MATHClassifier_QwenClassified()
    # custom or unknown: single bucket
    class _AllOne(_BasePromptClassifier):
        def classify(self, prompt_text: str, metadata: Optional[dict] = None) -> str:
            return "all"

    return _AllOne()


class ClassDroExp3p:
    def __init__(self, cfg: Exp3pConfig):
        self.cfg = cfg
        # Dynamic pass@k grouping classifier requires access to this instance state
        if str(cfg.prompt_classifier).lower() in {"passk_online", "acc_online", "online_passk"}:
            # Parse custom edges and extremes flag
            edges_txt = str(getattr(cfg, "passk_edges", "")).strip()
            edges_list = None
            if edges_txt:
                try:
                    if edges_txt.startswith("[") and edges_txt.endswith("]"):
                        edges_txt = edges_txt[1:-1]
                    edges_list = [float(x.strip()) for x in edges_txt.split(",") if x.strip()]
                except Exception:
                    edges_list = None
            exclude_extremes = bool(getattr(cfg, "passk_exclude_extremes", False))
            self.classifier = PassKOnlineClassifier(
                self,
                num_bins=int(getattr(cfg, "passk_num_bins", 10)),
                history_len=int(getattr(cfg, "passk_history_len", 0)),
                hysteresis=float(getattr(cfg, "passk_hysteresis", 0.0)),
                edges=edges_list,
                exclude_extremes=exclude_extremes,
            )
            self._dynamic_passk_enabled = True
        else:
            self.classifier = build_classifier(cfg.prompt_classifier)
            self._dynamic_passk_enabled = False
        self.cumulative_class_scores: Dict[str, float] = defaultdict(float)
        self.cumulative_class_counts: Dict[str, int] = defaultdict(int)
        # EMA of per-class step mean loss (used when cfg.debias_scores_ema=True)
        self.class_ema_scores: Dict[str, float] = {}
        # Cache of all-class statistics for z-scoring
        self._last_global_mean: float = 0.0
        self._last_global_std: float = 1.0
        # Online pass@k bookkeeping (uid-level)
        self._uid_seen_counts: Dict[str, int] = defaultdict(int)
        self._uid_correct_counts: Dict[str, int] = defaultdict(int)
        # Step counter for focus schedule
        self._step_updates: int = 0
        # Parse optional focus map
        self._focus_map: Dict[int, float] = {}
        try:
            if bool(getattr(cfg, "passk_focus_enable", False)) and isinstance(getattr(cfg, "passk_focus_map", ""), str):
                txt = str(getattr(cfg, "passk_focus_map", "")).strip()
                if txt:
                    # Accept optional brace-wrapped dict string
                    if txt.startswith("{") and txt.endswith("}"):
                        txt = txt[1:-1].strip()
                    for pair in txt.split(","):
                        pair = pair.strip()
                        if not pair:
                            continue
                        k, v = pair.split(":")
                        self._focus_map[int(k.strip())] = float(v.strip())
        except Exception:
            self._focus_map = {}
        # Optional rollout-arm state (Rollout-GDRO / Problem 2.1)
        self.rollout_arms: Optional[List[int]] = None
        self._rollout_arm_scores: Dict[str, np.ndarray] = {}
        self._rollout_arm_counts: Dict[str, np.ndarray] = {}

    def classify_batch(self, prompts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        class_ids: List[str] = []
        for i, p in enumerate(prompts):
            md = None
            if metadatas is not None and i < len(metadatas):
                md = metadatas[i]
            class_ids.append(self.classifier.classify(p, md))
        return class_ids

    def compute_weights(self, classes: List[str]) -> Dict[str, float]:
        # EXP3P-like exponential weights with exploration
        if not classes:
            return {}
        uniq = list(set(classes))
        exp_weights = {}
        # Prepare global normalization if requested
        if self.cfg.use_zscore:
            if self.cfg.debias_scores_ema and len(self.class_ema_scores) > 0:
                arr = np.array(list(self.class_ema_scores.values()), dtype=float)
            else:
                # Fall back to cumulative/count means across all seen classes
                keys = list(self.cumulative_class_scores.keys())
                vals = []
                for k in keys:
                    denom = float(max(1, self.cumulative_class_counts[k]))
                    vals.append(float(self.cumulative_class_scores[k]) / denom)
                arr = np.array(vals, dtype=float) if len(vals) > 0 else np.array([0.0], dtype=float)
            g_mean = float(arr.mean())
            g_std = float(arr.std())
            self._last_global_mean = g_mean
            self._last_global_std = g_std

        for c in uniq:
            score = float(self.cumulative_class_scores[c])
            # Prefer EMA if enabled
            if self.cfg.debias_scores_ema:
                score = float(self.class_ema_scores.get(c, 0.0))
            elif self.cfg.debias_scores:
                denom = float(max(1, self.cumulative_class_counts[c]))
                score = score / denom
            if self.cfg.use_zscore:
                denom = max(self.cfg.z_std_floor, float(self._last_global_std))
                z = (score - float(self._last_global_mean)) / denom
                z = float(np.clip(z, -self.cfg.z_cap, self.cfg.z_cap))
                score = z
            exp_weights[c] = float(np.exp(self.cfg.eta_q * score))
        total = sum(exp_weights.values())
        if total <= 0:
            normalized = {c: 1.0 / len(uniq) for c in uniq}
        else:
            normalized = {c: w / total for c, w in exp_weights.items()}
        final = {}
        for c in uniq:
            w = (1.0 - self.cfg.gamma) * normalized[c] + self.cfg.gamma / len(uniq)
            w_clipped = min(w, self.cfg.max_class_weight)
            # Optional focus mask schedule over acc bins
            if bool(getattr(self.cfg, "passk_focus_enable", False)) and self._focus_map:
                try:
                    bin_idx = None
                    if isinstance(c, str) and "accbin_" in c:
                        bin_idx = int(c.split("accbin_")[-1])
                    if bin_idx is not None and bin_idx in self._focus_map:
                        w_steps = int(getattr(self.cfg, "passk_focus_warmup_steps", 0))
                        r_steps = max(0, int(getattr(self.cfg, "passk_focus_ramp_steps", 0)))
                        if self._step_updates <= w_steps:
                            scale = 1.0
                        elif r_steps == 0:
                            scale = float(self._focus_map[bin_idx])
                        else:
                            t = (self._step_updates - w_steps) / float(r_steps)
                            if t < 0:
                                t = 0.0
                            elif t > 1.0:
                                t = 1.0
                            target = float(self._focus_map[bin_idx])
                            scale = 1.0 + t * (target - 1.0)
                        w_clipped = w_clipped * max(0.0, scale)
                except Exception:
                    pass
            final[c] = w_clipped
        return final

    # -----------------------------
    # Rollout arm utilities (Rollout-GDRO / Problem 2.1)
    # -----------------------------
    def set_rollout_arms(self, arms: List[int]):
        """Configure discrete rollout arms for Rollout-GDRO (Problem 2.1)."""
        try:
            uniq = sorted(set(int(a) for a in arms))
        except Exception:
            uniq = None
        if uniq and len(uniq) > 0:
            self.rollout_arms = uniq
            self._rollout_arm_scores = {}
            self._rollout_arm_counts = {}
        else:
            self.rollout_arms = None
            self._rollout_arm_scores = {}
            self._rollout_arm_counts = {}

    def _ensure_rollout_arm_state(self, cid: str):
        if self.rollout_arms is None:
            return
        if cid not in self._rollout_arm_scores:
            n = len(self.rollout_arms)
            self._rollout_arm_scores[cid] = np.zeros(n, dtype=float)
            self._rollout_arm_counts[cid] = np.zeros(n, dtype=float)

    def rollout_arm_probs(self, class_ids: List[str]) -> Dict[str, np.ndarray]:
        """Return per-class arm probabilities derived from EXP3P scores."""
        if not self.rollout_arms:
            return {}
        probs: Dict[str, np.ndarray] = {}
        uniq = set(class_ids)
        for cid in uniq:
            self._ensure_rollout_arm_state(cid)
            scores = self._rollout_arm_scores[cid]
            exp_scores = np.exp(self.cfg.eta_q * scores)
            total = float(exp_scores.sum())
            if total <= 0 or not np.isfinite(total):
                base = np.ones_like(exp_scores, dtype=float) / float(len(exp_scores))
            else:
                base = exp_scores / total
            p = (1.0 - self.cfg.gamma) * base + self.cfg.gamma / float(len(exp_scores))
            probs[cid] = p
        return probs

    def rollout_choose_arms(
        self,
        class_counts: Dict[str, int],
        total_budget: int,
        fallback_n: int,
    ) -> Tuple[Optional[Dict[str, int]], Optional[int]]:
        """Choose one rollout arm per class using a budget-constrained softmax over arms.

        Uses a DP over bins to pick the combination that maximizes summed log-probability
        while matching (or closest to) the target total_budget.
        """
        if not self.rollout_arms or not class_counts:
            return None, None
        probs = self.rollout_arm_probs(list(class_counts.keys()))
        arms = self.rollout_arms
        bin_order = list(class_counts.keys())
        # DP: sum -> (cost, choices)
        dp: Dict[int, Tuple[float, List[Tuple[str, int]]]] = {0: (0.0, [])}
        for cid in bin_order:
            cnt = int(class_counts.get(cid, 0))
            if cnt <= 0:
                continue
            p_vec = probs.get(cid)
            if p_vec is None or len(p_vec) != len(arms) or not np.all(np.isfinite(p_vec)):
                p_vec = np.ones(len(arms), dtype=float) / float(len(arms))
            logp = np.log(np.maximum(p_vec, 1e-12))
            new_dp: Dict[int, Tuple[float, List[Tuple[str, int]]]] = {}
            for acc_sum, (cost, choices) in dp.items():
                for ai, a in enumerate(arms):
                    contrib = a * cnt
                    ns = acc_sum + contrib
                    nc = cost - float(logp[ai])
                    prev = new_dp.get(ns, None)
                    if prev is None or nc < prev[0]:
                        new_dp[ns] = (nc, choices + [(cid, ai)])
            dp = new_dp
            if not dp:
                break
        if not dp:
            return None, None
        # Pick exact match if available, else closest by absolute diff then lowest cost
        best_sum = None
        best_cost = None
        for s, (c, _) in dp.items():
            if best_sum is None:
                best_sum, best_cost = s, c
            else:
                if abs(s - total_budget) < abs(best_sum - total_budget):
                    best_sum, best_cost = s, c
                elif abs(s - total_budget) == abs(best_sum - total_budget) and c < best_cost:
                    best_sum, best_cost = s, c
        if best_sum is None:
            return None, None
        chosen_pairs = dp[best_sum][1]
        result: Dict[str, int] = {}
        for cid, ai in chosen_pairs:
            result[cid] = int(arms[ai])
        # Fill any missing classes with fallback_n
        for cid in class_counts:
            result.setdefault(cid, int(fallback_n))
        return result, best_sum

    def update_rollout_arm_losses(
        self,
        class_n_map: Dict[str, int],
        expanded_class_ids: List[str],
        per_sample_lb: torch.Tensor,
        bar_n: float,
        mu: float,
    ):
        """Update arm scores using observed surrogate losses per bin."""
        if not self.rollout_arms:
            return
        if per_sample_lb is None or expanded_class_ids is None:
            return
        try:
            losses = per_sample_lb.detach().cpu().numpy().tolist()
        except Exception:
            return
        agg: Dict[str, List[float]] = defaultdict(list)
        for cid, lb in zip(expanded_class_ids, losses):
            agg[cid].append(float(lb))
        for cid, n_b in class_n_map.items():
            self._ensure_rollout_arm_state(cid)
            vals = agg.get(cid, None)
            if not vals:
                continue
            # per_sample_lb is a loss-like quantity (e.g. per-sample policy gradient loss).
            # To match Rollout-GDRO (Problem 2.1) inner minimization over J_b, treat J_b ≈ -mean(loss).
            mean_lb = float(np.mean(vals))
            est_j = -mean_lb
            adj_loss = est_j + mu * (float(n_b) - float(bar_n))
            try:
                arm_idx = self.rollout_arms.index(int(n_b))
            except ValueError:
                continue
            self._rollout_arm_scores[cid][arm_idx] += -adj_loss
            self._rollout_arm_counts[cid][arm_idx] += 1.0

    def weights_for_samples(self, class_ids: List[str]) -> torch.Tensor:
        cmap = self.compute_weights(class_ids)
        w = [cmap.get(c, 1.0) for c in class_ids]
        return torch.tensor(w, dtype=torch.float32)

    def update_with_losses(self, class_ids: List[str], per_sample_lb: torch.Tensor):
        # per_sample_lb should be detached CPU 1D tensor; we'll aggregate by class mean
        if per_sample_lb is None or len(class_ids) == 0:
            return
        # Advance focus schedule step counter
        self._step_updates += 1
        # Ensure numpy for robust handling
        losses = per_sample_lb.detach().cpu().numpy().tolist()
        by_class: Dict[str, List[float]] = defaultdict(list)
        for cid, lb in zip(class_ids, losses):
            by_class[cid].append(float(lb))
        total = sum(len(v) for v in by_class.values())
        for cid, lst in by_class.items():
            if len(lst) > 0:
                step_mean = float(np.mean(lst))
                if bool(getattr(self.cfg, "loss_norm_by_class", False)) and total > 0:
                    # Normalize by batch share to counter class prevalence
                    share = float(len(lst)) / float(total)
                    if share > 0:
                        step_mean = step_mean / share
                self.cumulative_class_scores[cid] += step_mean
                self.cumulative_class_counts[cid] += 1
                # Update EMA if enabled
                if self.cfg.debias_scores_ema:
                    if self.cumulative_class_counts[cid] <= 1 or cid not in self.class_ema_scores:
                        self.class_ema_scores[cid] = step_mean
                    else:
                        beta = float(self.cfg.ema_beta)
                        prev = float(self.class_ema_scores.get(cid, step_mean))
                        self.class_ema_scores[cid] = (1.0 - beta) * prev + beta * step_mean

    # -----------------------------
    # Dynamic pass@k updates (online)
    # -----------------------------
    def _get_uid_accuracy(self, uid: str) -> float:
        seen = int(self._uid_seen_counts.get(uid, 0))
        if seen <= 0:
            return 0.0
        correct = float(self._uid_correct_counts.get(uid, 0))
        return float(max(0.0, min(1.0, correct / max(1, seen))))

    def update_with_passk(self, uids: List[str], passk: List[float | int]):
        """Update running pass@k accuracy per uid.

        Args:
            uids: stable identifiers per original prompt (not per-response).
            passk: list of 0/1 indicating if any of k responses for the uid were correct at this step.
        """
        if not self._dynamic_passk_enabled:
            return
        if not uids or not passk:
            return
        for uid, ok in zip(uids, passk):
            try:
                key = str(uid)
                self._uid_seen_counts[key] += 1
                ok_val = 1.0 if float(ok) > 0.5 else 0.0
                if ok_val > 0.5:
                    self._uid_correct_counts[key] += 1
                # Also update classifier's _uid_recent deque if history_len > 0
                if hasattr(self, "classifier") and isinstance(self.classifier, PassKOnlineClassifier):
                    if self.classifier.history_len > 0:
                        if key not in self.classifier._uid_recent:
                            from collections import deque
                            self.classifier._uid_recent[key] = deque(
                                maxlen=self.classifier.history_len
                            )
                        self.classifier._uid_recent[key].append(ok_val)
            except Exception:
                # best-effort update; ignore malformed entries
                continue


