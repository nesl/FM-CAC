from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import yaml

from env_simulator import CarbonAwareEdgeEnv
from env_config import get_tier, available_tiers, TierConfig

HARDWARE_TIER: str = "heavy"


# Temporal slice of the grid trace (train / val / test / full).
@dataclass
class SplitInfo:
    name: str
    start_idx: int
    end_idx: int
    n_slots: int
    n_days: float
    g: np.ndarray
    p: np.ndarray


# Loads YAML scenario, resolves hardware tier, and builds envs per split.
class ExperimentConfig:

    VALID_SPLITS = ("train", "val", "test", "full")

    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            self.raw = yaml.safe_load(f)

        self.config_path = config_path

        if "hardware_tier" in self.raw:
            self.hardware_tier = self.raw["hardware_tier"]
        else:
            self.hardware_tier = HARDWARE_TIER
            print(
                f"  WARNING: 'hardware_tier' not found in {config_path}, "
                f"using default '{HARDWARE_TIER}'."
            )

        self._tier_cfg = get_tier(self.hardware_tier)
        self.MODELS = self._tier_cfg.MODELS
        self.HARDWARES = self._tier_cfg.HARDWARES
        self.HW_LABELS = self._tier_cfg.HW_LABELS

        self.data_cfg = self.raw["data"]
        self.split_cfg = self.raw["split"]
        self.episode_cfg = self.raw["episode"]
        self.battery_cfg = self.raw["battery"]
        self.qos_cfg = self.raw["qos"]
        self.reward_cfg = self.raw["reward"]
        self.workload_cfg = self.raw["workload"]
        self.norm_cfg = self.raw["normalization"]
        self.safety_cfg = self.raw["safety"]
        self.coverage_cfg = self.raw["coverage"]
        self.eval_cfg = self.raw["evaluation"]
        self.mpc_cfg = self.raw.get("mpc_policy", {})

        self.seeds: List[int] = list(self.raw["seeds"])
        assert len(self.seeds) >= 1, "Need at least one seed"

        ratios = (
            self.split_cfg["train_ratio"],
            self.split_cfg["val_ratio"],
            self.split_cfg["test_ratio"],
        )
        assert (
            abs(sum(ratios) - 1.0) < 1e-6
        ), f"Split ratios must sum to 1.0, got {sum(ratios)}: {ratios}"

        self._load_and_split()

    # Load CSV trace and chronologically partition into train/val/test/full.
    def _load_and_split(self):
        csv_path = self.data_cfg["csv_path"]
        ghg_col = self.data_cfg.get("ghg_col", "ghg_gCO2_per_kWh")
        price_col = self.data_cfg.get("price_col", "price_usd_per_kWh")
        delta_h = self.episode_cfg.get("delta_hours", 0.25)

        self.g_full, self.p_full = self._tier_cfg.load_traces_from_csv(
            csv_path,
            ghg_col=ghg_col,
            price_col=price_col,
            delta_hours=delta_h,
        )
        n = len(self.g_full)
        train_r = self.split_cfg["train_ratio"]
        val_r = self.split_cfg["val_ratio"]
        i_train_end = int(n * train_r)
        i_val_end = int(n * (train_r + val_r))

        self.splits: Dict[str, SplitInfo] = {
            "train": SplitInfo(
                "train",
                0,
                i_train_end,
                i_train_end,
                i_train_end * delta_h / 24.0,
                self.g_full[:i_train_end],
                self.p_full[:i_train_end],
            ),
            "val": SplitInfo(
                "val",
                i_train_end,
                i_val_end,
                i_val_end - i_train_end,
                (i_val_end - i_train_end) * delta_h / 24.0,
                self.g_full[i_train_end:i_val_end],
                self.p_full[i_train_end:i_val_end],
            ),
            "test": SplitInfo(
                "test",
                i_val_end,
                n,
                n - i_val_end,
                (n - i_val_end) * delta_h / 24.0,
                self.g_full[i_val_end:],
                self.p_full[i_val_end:],
            ),
            "full": SplitInfo(
                "full", 0, n, n, n * delta_h / 24.0, self.g_full.copy(), self.p_full.copy()
            ),
        }

    def get_split(self, split: str) -> SplitInfo:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from {self.VALID_SPLITS}")
        return self.splits[split]

    # Build a CarbonAwareEdgeEnv for the given split; YAML values fill defaults.
    def make_env(
        self,
        split: str = "train",
        seed: Optional[int] = None,
        horizon_T: Optional[int] = None,
        verbose: bool = False,
        w_acc: Optional[float] = None,
        w_carb_infer: Optional[float] = None,
        w_carb_charge: Optional[float] = None,
    ) -> CarbonAwareEdgeEnv:
        split_info = self.get_split(split)
        if seed is None:
            seed = self.seeds[0]
        if horizon_T is None:
            horizon_T = self.episode_cfg["horizon_T"]
        if split_info.n_slots < horizon_T:
            raise ValueError(
                f"Split '{split}' has {split_info.n_slots} slots but "
                f"horizon_T={horizon_T} requires at least that many."
            )

        delta_h = self.episode_cfg.get("delta_hours", 0.25)
        env = self._tier_cfg.make_env(
            g=split_info.g,
            p=split_info.p,
            horizon_T=horizon_T,
            delta_hours=delta_h,
            B_cap_mwh=self.battery_cfg["B_cap_mwh"],
            soc_min=self.battery_cfg["soc_min"],
            soc_max=self.battery_cfg["soc_max"],
            P_chg_mw=self.battery_cfg["P_chg_mw"],
            eta_chg=self.battery_cfg.get("eta_chg", 0.90),
            V_nom=self.battery_cfg["V_nom"],
            peukert_k=self.battery_cfg["peukert_k"],
            u_acc=self.qos_cfg["u_acc"],
            u_lat_s=self.qos_cfg["u_lat_s"],
            w_acc=w_acc if w_acc is not None else self.reward_cfg["w_acc"],
            w_carb_infer=(
                w_carb_infer if w_carb_infer is not None else self.reward_cfg["w_carb_infer"]
            ),
            w_carb_charge=(
                w_carb_charge if w_carb_charge is not None else self.reward_cfg["w_carb_charge"]
            ),
            infer_rate_ips=self.workload_cfg["infer_rate_ips"],
            camera_fps=self.workload_cfg["camera_fps"],
            normalize_costs=self.norm_cfg["normalize_costs"],
            power_utilization=self.norm_cfg["power_utilization"],
            g_ref_max=float(np.max(self.g_full)),
            invalid_action_penalty=self.safety_cfg["invalid_action_penalty"],
            terminate_on_invalid=self.safety_cfg["terminate_on_invalid"],
            disallow_charge_when_full=self.safety_cfg["disallow_charge_when_full"],
            use_coverage=self.coverage_cfg["use_coverage"],
            seed=seed,
            verbose=verbose,
        )

        return env

    # Aggregate per-seed episode metrics into seed-mean stats (mean/std/min/max).
    @staticmethod
    def aggregate_results(all_seed_results: Dict[int, dict]) -> dict:
        seeds = sorted(all_seed_results.keys())

        def _seed_means(key):
            return np.array([float(np.mean(all_seed_results[s][key])) for s in seeds])

        def _seed_means_from_concat(key):
            vals = []
            for s in seeds:
                per_ep = all_seed_results[s].get(key, [])
                if len(per_ep) > 0:

                    all_vals = (
                        np.concatenate(per_ep) if isinstance(per_ep[0], np.ndarray) else per_ep
                    )
                    vals.append(float(np.mean(all_vals)) if len(all_vals) > 0 else 0.0)
                else:
                    vals.append(0.0)
            return np.array(vals)

        agg = {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "per_seed_reward": _seed_means("rewards"),
            "per_seed_carbon": _seed_means("carbons"),
            "per_seed_cost": _seed_means("costs"),
            "per_seed_accuracy": _seed_means("mean_accuracy"),
            "per_seed_latency": _seed_means("mean_latency"),
            "per_seed_invalid_actions": _seed_means("invalid_actions"),
            "per_seed_mode_switches": _seed_means("mode_switches"),
            "per_seed_charge_rate": _seed_means("charge_rates"),
            "per_seed_battery_pct": _seed_means("battery_pct"),
            "per_seed_grid_pct": _seed_means("grid_pct"),
            "per_seed_ci_when_charging": _seed_means_from_concat("grid_ci_when_charging"),
            "per_seed_ci_when_not_charging": _seed_means_from_concat("grid_ci_when_not_charging"),
            "per_seed_price_when_charging": _seed_means_from_concat("grid_price_when_charging"),
            "per_seed_price_when_not_charging": _seed_means_from_concat(
                "grid_price_when_not_charging"
            ),
        }

        for key_base in [
            "reward",
            "carbon",
            "cost",
            "accuracy",
            "latency",
            "invalid_actions",
            "mode_switches",
            "charge_rate",
            "battery_pct",
            "grid_pct",
            "ci_when_charging",
            "ci_when_not_charging",
            "price_when_charging",
            "price_when_not_charging",
        ]:
            arr = agg[f"per_seed_{key_base}"]
            agg[f"{key_base}_mean"] = float(np.mean(arr))
            agg[f"{key_base}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        return agg

    # Pretty-print the aggregated cross-seed table to stdout.
    @staticmethod
    def print_cross_seed_summary(agg: dict, algo_name: str = "Algorithm", split_name: str = "test"):
        n = agg["n_seeds"]
        print(f"\n{'='*75}")
        print(f"  {algo_name} — Cross-Seed Summary ({n} seeds, {split_name} split)")
        print(f"{'='*75}")

        def _print_table(title, rows):
            print(f"\n  {title}:")
            print(f"  {'Metric':<30} {'Mean':>12} {'± Std':>12}")
            print(f"  {'-'*56}")
            for label, key, fmt in rows:
                m = f"{agg[f'{key}_mean']:{fmt}}"
                s = f"{agg[f'{key}_std']:{fmt}}"
                print(f"  {label:<30} {m:>12} {s:>12}")

        _print_table(
            "PRIMARY METRICS (Eq. 13 objective terms)",
            [
                ("Reward (total)", "reward", "+.2f"),
                ("Carbon (gCO₂)", "carbon", ".4f"),
                ("Cost ($)", "cost", ".6f"),
            ],
        )

        _print_table(
            "QoS METRICS",
            [
                ("Accuracy (mAP)", "accuracy", ".4f"),
                ("Latency (ms)", "latency", ".2f"),
                ("Invalid actions", "invalid_actions", ".1f"),
            ],
        )

        _print_table(
            "BEHAVIORAL METRICS",
            [
                ("Battery source (%)", "battery_pct", ".1f"),
                ("Grid source (%)", "grid_pct", ".1f"),
                ("Charge rate (%)", "charge_rate", ".1f"),
                ("Mode switches/ep", "mode_switches", ".1f"),
            ],
        )

        _print_table(
            "CARBON-AWARENESS (key paper insight)",
            [
                ("CI when charging", "ci_when_charging", ".1f"),
                ("CI when NOT charging", "ci_when_not_charging", ".1f"),
                ("Price when charging", "price_when_charging", ".4f"),
                ("Price when NOT chg", "price_when_not_charging", ".4f"),
            ],
        )

        ci_chg = agg["ci_when_charging_mean"]
        ci_nochg = agg["ci_when_not_charging_mean"]
        if ci_nochg > 1e-6:
            pct = (ci_nochg - ci_chg) / ci_nochg * 100
            print(f"\n  → Charges at {pct:.1f}% lower CI than non-charging slots")

        print(f"{'='*75}")

    # Dump aggregated + per-seed results to a JSON file.
    @staticmethod
    def save_results_json(
        agg: dict, all_seed_results: Dict[int, dict], save_path: str, algo_name: str = "Algorithm"
    ):

        def _safe_float(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        out = {
            "algorithm": algo_name,
            "n_seeds": agg["n_seeds"],
            "seeds": agg["seeds"],
            "primary": {
                "reward": {"mean": agg["reward_mean"], "std": agg["reward_std"]},
                "carbon": {"mean": agg["carbon_mean"], "std": agg["carbon_std"]},
                "cost": {"mean": agg["cost_mean"], "std": agg["cost_std"]},
            },
            "qos": {
                "accuracy": {"mean": agg["accuracy_mean"], "std": agg["accuracy_std"]},
                "latency_ms": {"mean": agg["latency_mean"], "std": agg["latency_std"]},
                "invalid_actions": {
                    "mean": agg["invalid_actions_mean"],
                    "std": agg["invalid_actions_std"],
                },
            },
            "behavior": {
                "battery_pct": {"mean": agg["battery_pct_mean"], "std": agg["battery_pct_std"]},
                "grid_pct": {"mean": agg["grid_pct_mean"], "std": agg["grid_pct_std"]},
                "charge_rate": {"mean": agg["charge_rate_mean"], "std": agg["charge_rate_std"]},
                "mode_switches": {
                    "mean": agg["mode_switches_mean"],
                    "std": agg["mode_switches_std"],
                },
            },
            "carbon_awareness": {
                "ci_when_charging": {
                    "mean": agg["ci_when_charging_mean"],
                    "std": agg["ci_when_charging_std"],
                },
                "ci_when_not_charging": {
                    "mean": agg["ci_when_not_charging_mean"],
                    "std": agg["ci_when_not_charging_std"],
                },
                "price_when_charging": {
                    "mean": agg["price_when_charging_mean"],
                    "std": agg["price_when_charging_std"],
                },
                "price_when_not_charging": {
                    "mean": agg["price_when_not_charging_mean"],
                    "std": agg["price_when_not_charging_std"],
                },
            },
            "per_seed": {},
        }

        for s in agg["seeds"]:
            r = all_seed_results[s]
            out["per_seed"][str(s)] = {
                "reward_mean": _safe_float(np.mean(r["rewards"])),
                "carbon_mean": _safe_float(np.mean(r["carbons"])),
                "cost_mean": _safe_float(np.mean(r["costs"])),
                "accuracy_mean": _safe_float(np.mean(r["mean_accuracy"])),
                "latency_mean": _safe_float(np.mean(r["mean_latency"])),
                "mode_switches": _safe_float(np.mean(r["mode_switches"])),
                "charge_rate": _safe_float(np.mean(r["charge_rates"])),
                "battery_pct": _safe_float(np.mean(r["battery_pct"])),
            }

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2, default=_safe_float)
        print(f"  Results JSON -> {save_path}")

    # Echo the resolved config to stdout for run reproducibility.
    def print_summary(self):
        delta_h = self.episode_cfg.get("delta_hours", 0.25)
        print("=" * 80)
        print("  Experiment Configuration")
        print("=" * 80)
        print(f"  Config file : {self.config_path}")
        print(f"  Hardware    : {self.hardware_tier} ({self._tier_cfg.desc})")
        print(f"  CSV data    : {self.data_cfg['csv_path']}")
        print(
            f"  Total slots : {len(self.g_full):,} " f"({len(self.g_full) * delta_h / 24:.0f} days)"
        )
        print(
            f"  Horizon     : {self.episode_cfg['horizon_T']} slots "
            f"({self.episode_cfg['horizon_T'] * delta_h / 24:.0f} days)"
        )
        print()
        print("  Data Splits (temporal, no overlap):")
        print(
            f"  {'Split':<8} {'Ratio':>6} {'Slots':>10} {'Days':>8} "
            f"{'GHG mean':>10} {'Price mean':>12}"
        )
        print(f"  {'-'*60}")
        for name in ("train", "val", "test"):
            s = self.splits[name]
            print(
                f"  {name:<8} {self.split_cfg[f'{name}_ratio']:>5.0%} "
                f"{s.n_slots:>10,} {s.n_days:>7.0f}d "
                f"{np.mean(s.g):>9.1f}   ${np.mean(s.p):>10.4f}"
            )
        print()
        print(
            f"  Battery     : {self.battery_cfg['B_cap_mwh']/1000:.0f} Wh, "
            f"SoC=[{self.battery_cfg['soc_min']:.0%}, {self.battery_cfg['soc_max']:.0%}]"
        )
        print(
            f"  QoS         : acc >= {self.qos_cfg['u_acc']}, "
            f"lat <= {self.qos_cfg['u_lat_s']*1000:.0f} ms"
        )
        print(
            f"  Reward wts  : acc={self.reward_cfg['w_acc']}, "
            f"carb_infer={self.reward_cfg['w_carb_infer']}, "
            f"carb_charge={self.reward_cfg['w_carb_charge']}"
        )
        print(f"  Seeds       : {self.seeds} ({len(self.seeds)} runs)")
        print(f"  Eval episodes: {self.eval_cfg['eval_episodes']} per seed")
        print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test config loader")
    parser.add_argument(
        "--config", default="./configs/exp_heavy.yaml", help="Path to experiment config YAML"
    )
    args = parser.parse_args()

    cfg = ExperimentConfig(args.config)
    cfg.print_summary()

    for split in ("train", "val", "test"):
        print(f"\n--- Building {split} env (seed={cfg.seeds[0]}) ---")
        env = cfg.make_env(split, seed=cfg.seeds[0], verbose=True)
        obs, info = env.reset()
        print(
            f"  {split}: {cfg.splits[split].n_slots:,} slots, "
            f"obs={obs}, mask sum={info['action_mask'].sum()}"
        )

    print("\nAll splits OK.")
