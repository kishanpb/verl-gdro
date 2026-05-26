# verl-gdro

**Paper:** *Group Distributionally Robust Optimization-Driven Reinforcement Learning for LLM Reasoning*
**Venue:** ICML 2026

This repository contains a patch-style subset of files implementing both **Prompt-GDRO** and **Rollout-GDRO**. It is intended to be applied on top of the base VERL commit:

https://github.com/volcengine/verl/commit/bd756c15c8873e40cfbb5ef2a4d1d6cadb4ea272

## Structure

```
verl/
├── trainer/
│   ├── group_dro_grpo.py          # Core GDRO controller (EXP3P weights, online classifier)
│   ├── config/algorithm.py         # Algorithm config extensions
│   └── ppo/ray_trainer.py          # Modified trainer with GDRO integration
├── utils/
│   ├── budget_allocation.py        # Rollout-GDRO variance-aware budget allocation
│   ├── debug/metrics.py            # Debug metric utilities
│   └── reward_score/               # Reward scoring (math, GPQA)
└── workers/
    ├── actor/dp_actor.py           # Actor with variable rollout support
    └── reward_manager/naive.py     # Reward manager
examples/
└── data_preprocess/                # Dataset preparation scripts
algorithm_comparison/
├── run_groupdro.sh                 # Prompt-GDRO launch script
└── run_rollout_groupdro.sh         # Rollout-GDRO launch script
```

## Usage

1. Clone the base VERL repository at the commit above.
2. Copy/overlay the files from this patch onto the VERL tree.
3. Use the scripts in `algorithm_comparison/` to launch training runs.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{panaganti2026gdro,
  title={Group Distributionally Robust Optimization-Driven Reinforcement Learning for LLM Reasoning},
  author={Panaganti, Kishan and Liang, Zhenwen and Yu, Wenhao and Mi, Haitao and Yu, Dong},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
