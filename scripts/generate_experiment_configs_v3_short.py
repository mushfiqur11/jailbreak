#!/usr/bin/env python3
"""
Generate SHORT V3 experiment configuration files for a 5-run chained workflow.

This short generator is identical to V3 main, except:
  - max_goals = 3

Used to validate end-to-end wiring quickly.
"""

import os

from generate_experiment_configs_v3 import generate_experiment_configs


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "configs", "experiments_v3_short")

    names = generate_experiment_configs(
        output_dir=out_dir,
        max_goals=3,
        exp_prefix="short_v3",
    )

    list_path = os.path.join(out_dir, "experiment_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")

    print(f"Generated {len(names)} SHORT V3 experiments in: {out_dir}")
    print(f"Experiment list: {list_path}")


if __name__ == "__main__":
    main()
