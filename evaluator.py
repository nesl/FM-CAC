#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import json
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_loader import ExperimentConfig
from env_simulator import CarbonAwareEdgeEnv


# Minimal policy interface: act(obs, mask) -> flat action id.
class BasePolicy(ABC):

    @abstractmethod
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:

        pass

    def reset_episode(self) -> None:
        pass


# Roll out one episode; returns per-step history for downstream metrics/plots.
def run_episode(
    env: CarbonAwareEdgeEnv,
    policy: BasePolicy,
    ep_start: int,
) -> dict:
    obs, info = env.reset(options={"ep_start": ep_start})
    mask = info["action_mask"]
    policy.reset_episode()

    data = {
        "step": [],
        "battery_mwh": [],
        "grid_ci": [],
        "grid_price": [],
        "bat_avg_ci": [],
        "bat_avg_price": [],
        "mode_index": [],
        "mode_name": [],
        "charge": [],
        "source": [],
        "reward": [],
        "carbon_g": [],
        "cost_usd": [],
        "U_acc": [],
        "E_charge_mwh": [],
        "E_infer_mwh": [],
        "N_infer": [],
        "invalid": [],
        "soc_pct": [],
    }

    done = False
    t = 0
    while not done:
        flat_action = policy.act(obs, mask)

        src = flat_action % 2
        rem = flat_action // 2
        chg = rem % 2
        mode = rem // 2
        raw_action = np.array([mode, chg, src], dtype=int)

        obs_next, reward, terminated, truncated, info = env.step(raw_action)
        done = terminated or truncated

        data["step"].append(t)
        data["battery_mwh"].append(float(obs[0]))
        data["grid_ci"].append(float(obs[1]))
        data["grid_price"].append(float(obs[2]))
        data["bat_avg_ci"].append(float(obs[3]))
        data["bat_avg_price"].append(float(obs[4]))
        data["mode_index"].append(mode)
        data["charge"].append(chg)
        data["source"].append(src)
        data["reward"].append(float(reward))
        data["invalid"].append(info.get("invalid_action", False))
        data["carbon_g"].append(info.get("total_carbon_g", 0.0))
        data["cost_usd"].append(info.get("total_cost_usd", 0.0))
        data["U_acc"].append(info.get("U_acc", 0.0))
        data["E_charge_mwh"].append(info.get("E_charge_accepted_mwh", 0.0))
        data["E_infer_mwh"].append(info.get("E_infer_raw_mwh", 0.0))
        data["N_infer"].append(info.get("N_infer", 0))

        data["soc_pct"].append(
            (float(obs[0]) - env.B_min_mwh) / max(env.B_usable_mwh, 1e-12) * 100.0
        )

        pair = env.mode_id_to_pair.get(mode, None)
        if pair is not None:
            n_id, h_id = pair
            data["mode_name"].append(f"{env.models[n_id].name}@hw{h_id}")
        else:
            data["mode_name"].append("idle")

        obs = obs_next
        mask = info.get("action_mask", env.get_action_mask())
        t += 1

    data["battery_mwh_terminal"] = float(obs[0])

    for k in data:
        if k != "mode_name":
            data[k] = np.array(data[k])

    return data


# Reduce per-step history into scalar episode metrics (reward, carbon, cost, etc.).
def extract_episode_metrics(data: dict, env: CarbonAwareEdgeEnv) -> dict:
    valid = ~data["invalid"]
    T = len(data["step"])

    reward_total = float(data["reward"].sum())
    carbon_total = float(data["carbon_g"][valid].sum())
    cost_total = float(data["cost_usd"][valid].sum())

    accuracies = []
    latencies = []
    for m in data["mode_index"][valid]:
        pair = env.mode_id_to_pair.get(int(m), None)
        if pair is not None:
            attr = env.attr_table[pair]
            accuracies.append(attr.alpha)
            latencies.append(attr.latency_s_per_infer * 1000.0)
    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    n_invalid = int(data["invalid"].sum())

    valid_modes = data["mode_index"][valid]
    mode_switches = int(np.sum(valid_modes[1:] != valid_modes[:-1])) if len(valid_modes) > 1 else 0

    if valid.any():
        charge_rate = float(data["charge"][valid].mean() * 100.0)
        battery_pct = float((data["source"][valid] == 0).mean() * 100.0)
        grid_pct = float((data["source"][valid] == 1).mean() * 100.0)
    else:
        charge_rate = 0.0
        battery_pct = 0.0
        grid_pct = 0.0

    charging_mask = data["charge"].astype(bool) & valid
    not_charging_mask = (~data["charge"].astype(bool)) & valid

    ci_when_charging = data["grid_ci"][charging_mask] if charging_mask.any() else np.array([])
    ci_when_not_charging = (
        data["grid_ci"][not_charging_mask] if not_charging_mask.any() else np.array([])
    )
    price_when_charging = data["grid_price"][charging_mask] if charging_mask.any() else np.array([])
    price_when_not_charging = (
        data["grid_price"][not_charging_mask] if not_charging_mask.any() else np.array([])
    )

    return {
        "reward_total": reward_total,
        "carbon_total": carbon_total,
        "cost_total": cost_total,
        "mean_accuracy": mean_acc,
        "mean_latency": mean_lat,
        "invalid_actions": n_invalid,
        "mode_switches": mode_switches,
        "charge_rate": charge_rate,
        "battery_pct": battery_pct,
        "grid_pct": grid_pct,
        "ci_when_charging": ci_when_charging,
        "ci_when_not_charging": ci_when_not_charging,
        "price_when_charging": price_when_charging,
        "price_when_not_charging": price_when_not_charging,
    }


# Transpose list-of-episode-dicts into dict-of-lists for cross-seed aggregation.
def collect_seed_results(all_episode_metrics: List[dict]) -> dict:
    return {
        "rewards": [m["reward_total"] for m in all_episode_metrics],
        "carbons": [m["carbon_total"] for m in all_episode_metrics],
        "costs": [m["cost_total"] for m in all_episode_metrics],
        "mean_accuracy": [m["mean_accuracy"] for m in all_episode_metrics],
        "mean_latency": [m["mean_latency"] for m in all_episode_metrics],
        "invalid_actions": [m["invalid_actions"] for m in all_episode_metrics],
        "mode_switches": [m["mode_switches"] for m in all_episode_metrics],
        "charge_rates": [m["charge_rate"] for m in all_episode_metrics],
        "battery_pct": [m["battery_pct"] for m in all_episode_metrics],
        "grid_pct": [m["grid_pct"] for m in all_episode_metrics],
        "grid_ci_when_charging": [m["ci_when_charging"] for m in all_episode_metrics],
        "grid_ci_when_not_charging": [m["ci_when_not_charging"] for m in all_episode_metrics],
        "grid_price_when_charging": [m["price_when_charging"] for m in all_episode_metrics],
        "grid_price_when_not_charging": [m["price_when_not_charging"] for m in all_episode_metrics],
    }


# Tile a split into non-overlapping (or overlapping) episode start indices.
def compute_sequential_ep_starts(
    split_n_slots: int,
    horizon_T: int,
    overlap: float = 0.0,
):
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")

    stride = max(1, int(round(horizon_T * (1.0 - overlap))))

    max_start = split_n_slots - horizon_T
    if max_start < 0:
        raise ValueError(
            f"Split has {split_n_slots} slots but horizon_T={horizon_T} "
            f"requires at least {horizon_T}. Cannot run any episodes."
        )

    ep_starts = list(range(0, max_start + 1, stride))

    if len(ep_starts) == 0:
        raise ValueError(
            f"No episodes could be generated with split_n_slots={split_n_slots}, "
            f"horizon_T={horizon_T}, overlap={overlap} (stride={stride})."
        )

    last_ep_end = ep_starts[-1] + horizon_T
    leftover = split_n_slots - last_ep_end

    return ep_starts, stride, leftover


# Pretty-print one episode's metrics block to stdout.
def print_episode_summary(
    data: dict,
    env: CarbonAwareEdgeEnv,
    algo_name: str,
    ep_idx: int = 0,
    ep_start: int = 0,
) -> None:
    T = len(data["step"])
    delta_h = env.cfg.delta_hours

    print(f"\n{'='*70}")
    print(f"  {algo_name} — Episode {ep_idx} Summary")
    print(
        f"  (data offset: slot {ep_start} — {ep_start + T}, "
        f"covering {T * delta_h / 24:.1f} days)"
    )
    print(f"{'='*70}")

    print(f"\n  Steps: {T}  (expected {env.cfg.horizon_T})")
    print(f"  Invalid actions: {int(data['invalid'].sum())}")

    valid = ~data["invalid"]

    print(f"\n  REWARDS:")
    print(f"    Total reward : {data['reward'].sum():+.4f}")
    print(f"    Mean/step    : {data['reward'].mean():+.6f}")
    print(f"    Min/Max      : {data['reward'].min():+.6f} / {data['reward'].max():+.6f}")

    print(f"\n  CARBON:")
    print(f"    Total carbon : {data['carbon_g'][valid].sum():.6f} gCO2")
    print(f"    Mean/step    : {data['carbon_g'][valid].mean():.6f} gCO2")

    print(f"\n  COST:")
    print(f"    Total cost   : ${data['cost_usd'][valid].sum():.8f}")
    print(f"    Mean/step    : ${data['cost_usd'][valid].mean():.8f}")

    print(f"\n  PERFORMANCE:")
    print(f"    Mean U_acc  : {data['U_acc'][valid].mean():.6f}")
    print(f"    Mean N_infer : {data['N_infer'][valid].mean():.1f} inferences/slot")

    print(f"\n  BATTERY:")
    print(f"    Start        : {data['battery_mwh'][0]:.0f} mWh")
    print(f"    End          : {data['battery_mwh_terminal']:.0f} mWh")
    print(
        f"    Min / Max    : {data['battery_mwh'].min():.0f} / {data['battery_mwh'].max():.0f} mWh"
    )
    print(f"    SoC range    : [{env.B_min_mwh:.0f}, {env.B_max_mwh:.0f}] mWh")

    print(f"\n  CHARGING:")
    n_charge = int(data["charge"][valid].sum())
    n_valid = int(valid.sum())
    if n_valid > 0:
        print(f"    Charge steps : {n_charge} / {n_valid} " f"({n_charge/n_valid*100:.1f}%)")
        print(f"    Total charged: {data['E_charge_mwh'][valid].sum():.1f} mWh")
    else:
        print(f"    Charge steps : 0 / 0 (n/a — all steps invalid)")
        print(f"    Total charged: 0.0 mWh")

    print(f"\n  SOURCE:")
    if n_valid > 0:
        n_bat = int((data["source"][valid] == 0).sum())
        n_grid = int((data["source"][valid] == 1).sum())
        print(f"    Battery steps: {n_bat} ({n_bat/n_valid*100:.1f}%)")
        print(f"    Grid steps   : {n_grid} ({n_grid/n_valid*100:.1f}%)")
    else:
        print(f"    Battery steps: 0 (n/a — all steps invalid)")
        print(f"    Grid steps   : 0 (n/a — all steps invalid)")

    print(f"\n  MODES USED:")
    unique_modes, counts = np.unique(data["mode_index"], return_counts=True)
    for m, c in sorted(zip(unique_modes, counts), key=lambda x: -x[1]):
        pair = env.mode_id_to_pair.get(int(m), None)
        if pair:
            n_id, h_id = pair
            name = f"{env.models[n_id].name} @ hw{h_id}"
            alpha = env.attr_table[pair].alpha
            lat = env.attr_table[pair].latency_s_per_infer * 1000
            print(
                f"    Mode {m:2d}: {name:<30} α={alpha:.3f} lat={lat:.1f}ms  "
                f"× {c} steps ({c/T*100:.1f}%)"
            )
        else:
            print(f"    Mode {m:2d}: idle  × {c} steps ({c/T*100:.1f}%)")

    print(f"\n  GRID SIGNALS:")
    print(
        f"    CI  : mean={data['grid_ci'].mean():.1f}  "
        f"min={data['grid_ci'].min():.1f}  max={data['grid_ci'].max():.1f} gCO2/kWh"
    )
    print(
        f"    Price: mean={data['grid_price'].mean():.4f}  "
        f"min={data['grid_price'].min():.4f}  max={data['grid_price'].max():.4f} $/kWh"
    )

    print(f"{'='*70}")


# Save a multi-panel trajectory PNG for one episode (CI, battery, action, etc.).
def plot_episode(
    data: dict,
    env: CarbonAwareEdgeEnv,
    save_dir: str,
    algo_name: str,
    episode_idx: int = 0,
    ep_start: int = 0,
) -> None:
    T = len(data["step"])
    steps = data["step"]
    hours = steps * env.cfg.delta_hours

    fig, axes = plt.subplots(7, 1, figsize=(16, 22), sharex=True)
    fig.suptitle(
        f"{algo_name} — Episode {episode_idx} "
        f"(slots {ep_start}–{ep_start+T}, {T} steps, "
        f"{T * env.cfg.delta_hours / 24:.0f} days)",
        fontsize=14,
        fontweight="bold",
    )

    hours_with_terminal = np.append(hours, hours[-1] + env.cfg.delta_hours)
    battery_with_terminal = np.append(data["battery_mwh"], data["battery_mwh_terminal"])
    ax = axes[0]
    ax.fill_between(hours_with_terminal, battery_with_terminal, alpha=0.3, color="tab:blue")
    ax.plot(hours_with_terminal, battery_with_terminal, color="tab:blue", lw=1)
    ax.axhline(
        env.B_min_mwh, color="red", ls="--", lw=0.8, label=f"SoC floor ({env.B_min_mwh:.0f})"
    )
    ax.axhline(
        env.B_max_mwh, color="green", ls="--", lw=0.8, label=f"SoC ceiling ({env.B_max_mwh:.0f})"
    )
    ax.set_ylabel("Battery (mWh)")
    ax.set_title("Battery State of Charge")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(hours, data["grid_ci"], color="tab:red", lw=0.8, label="CI (gCO₂/kWh)")
    ax.set_ylabel("CI (gCO₂/kWh)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    ax2.plot(hours, data["grid_price"], color="tab:green", lw=0.8, label="Price ($/kWh)")
    ax2.set_ylabel("Price ($/kWh)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax.set_title("Grid Signals")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax = axes[2]
    ax.fill_between(hours, data["charge"], step="post", alpha=0.4, color="tab:orange")
    ax.set_ylabel("Charge (0/1)")
    ax.set_ylim(-0.1, 1.1)
    valid_plot = ~data["invalid"]
    n_valid_plot = int(valid_plot.sum())
    charge_rate_valid = (data["charge"][valid_plot].mean() * 100) if n_valid_plot > 0 else 0.0
    ax.set_title(f"Charging Decision (valid-step rate: {charge_rate_valid:.1f}%)")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.fill_between(hours, data["source"], step="post", alpha=0.4, color="tab:purple")
    ax.set_ylabel("Source")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Battery", "Grid"])
    n_bat_valid = int((data["source"][valid_plot] == 0).sum()) if n_valid_plot > 0 else 0
    n_grid_valid = n_valid_plot - n_bat_valid
    bat_pct = n_bat_valid / n_valid_plot * 100 if n_valid_plot > 0 else 0.0
    grid_pct = n_grid_valid / n_valid_plot * 100 if n_valid_plot > 0 else 0.0
    ax.set_title(f"Power Source (battery: {bat_pct:.1f}%, grid: {grid_pct:.1f}%, valid steps)")
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.scatter(hours, data["mode_index"], s=2, alpha=0.5, color="tab:brown")
    ax.set_ylabel("Mode Index")
    ax.set_title("Operating Mode Selected")
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    ax.plot(hours, data["reward"], color="tab:blue", lw=0.5, alpha=0.5)
    window = min(96, T // 4)
    if window > 1:
        rolling = np.convolve(data["reward"], np.ones(window) / window, mode="valid")
        ax.plot(
            hours[: len(rolling)],
            rolling,
            color="tab:blue",
            lw=1.5,
            label=f"Rolling avg ({window} steps)",
        )
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_ylabel("Reward")
    ax.set_title(f"Per-Step Reward (total: {data['reward'].sum():+.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[6]
    valid = ~data["invalid"]
    ax.bar(
        hours[valid],
        data["carbon_g"][valid],
        width=env.cfg.delta_hours * 0.8,
        alpha=0.5,
        color="tab:red",
        label="Carbon (gCO₂)",
    )
    ax.set_ylabel("Carbon (gCO₂)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    ax2.bar(
        hours[valid] + env.cfg.delta_hours * 0.3,
        data["cost_usd"][valid] * 1000,
        width=env.cfg.delta_hours * 0.4,
        alpha=0.5,
        color="tab:green",
        label="Cost (m$)",
    )
    ax2.set_ylabel("Cost (m$)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax.set_xlabel("Time (hours)")
    ax.set_title(
        f"Carbon & Cost (total: {data['carbon_g'][valid].sum():.4f} gCO₂, "
        f"${data['cost_usd'][valid].sum():.6f})"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{algo_name.lower()}_ep{episode_idx}.png"
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# Save a single summary PNG overlaying trajectories from all episodes.
def plot_multi_episode_summary(
    all_data: List[dict],
    save_dir: str,
    algo_name: str,
    tiling_label: str = "sequential",
) -> None:
    n_ep = len(all_data)
    if n_ep < 2:
        return

    rewards = [d["reward"].sum() for d in all_data]
    carbons = [d["carbon_g"][~d["invalid"]].sum() for d in all_data]
    costs = [d["cost_usd"][~d["invalid"]].sum() for d in all_data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(range(n_ep), rewards, color="tab:blue", alpha=0.7)
    axes[0].set_title(f"Reward (mean={np.mean(rewards):+.2f} ± {np.std(rewards):.2f})")
    axes[0].set_xlabel(f"Episode ({tiling_label})")
    axes[0].axhline(np.mean(rewards), color="red", ls="--", lw=1)

    axes[1].bar(range(n_ep), carbons, color="tab:red", alpha=0.7)
    axes[1].set_title(f"Carbon gCO₂ (mean={np.mean(carbons):.4f} ± {np.std(carbons):.4f})")
    axes[1].set_xlabel(f"Episode ({tiling_label})")
    axes[1].axhline(np.mean(carbons), color="red", ls="--", lw=1)

    axes[2].bar(range(n_ep), costs, color="tab:green", alpha=0.7)
    axes[2].set_title(f"Cost $ (mean={np.mean(costs):.6f} ± {np.std(costs):.6f})")
    axes[2].set_xlabel(f"Episode ({tiling_label})")
    axes[2].axhline(np.mean(costs), color="red", ls="--", lw=1)

    plt.suptitle(
        f"{algo_name} — {n_ep} Episodes ({tiling_label}, full split)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fname = f"{algo_name.lower()}_summary.png"
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Dump the per-step episode trace to CSV for downstream analysis.
def save_episode_csv(
    data: dict,
    save_dir: str,
    algo_name: str,
    episode_idx: int = 0,
) -> None:
    fname = f"{algo_name.lower()}_ep{episode_idx}.csv"
    path = os.path.join(save_dir, fname)
    columns = [
        "step",
        "battery_mwh",
        "soc_pct",
        "grid_ci",
        "grid_price",
        "bat_avg_ci",
        "bat_avg_price",
        "mode_index",
        "mode_name",
        "charge",
        "source",
        "reward",
        "carbon_g",
        "cost_usd",
        "U_acc",
        "E_charge_mwh",
        "E_infer_mwh",
        "N_infer",
        "invalid",
    ]
    T = len(data["step"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(T):
            row = []
            for col in columns:
                val = data[col][i] if col != "mode_name" else data["mode_name"][i]
                if isinstance(val, (np.bool_, bool)):
                    row.append(int(val))
                elif isinstance(val, (np.floating, float)):
                    row.append(f"{val:.8g}")
                else:
                    row.append(val)
            writer.writerow(row)
    print(f"  Saved CSV: {path}  ({T} rows)")


# Aggregate episode metrics across one seed's episodes and persist to JSON.
def save_seed_summary_json(
    all_data: List[dict],
    all_ep_starts: List[int],
    env: CarbonAwareEdgeEnv,
    save_dir: str,
    algo_name: str,
    split_name: str,
    seed: int,
    overlap: float = 0.0,
    stride: int = 0,
) -> dict:
    n_ep = len(all_data)

    ep_stats = []
    for i, d in enumerate(all_data):
        valid = ~d["invalid"]
        ep = {
            "episode": i,
            "ep_start_slot": int(all_ep_starts[i]),
            "n_steps": int(len(d["step"])),
            "n_invalid": int(d["invalid"].sum()),
            "reward_total": float(d["reward"].sum()),
            "reward_mean": float(d["reward"].mean()),
            "reward_min": float(d["reward"].min()),
            "reward_max": float(d["reward"].max()),
            "carbon_total_g": float(d["carbon_g"][valid].sum()),
            "carbon_mean_g": float(d["carbon_g"][valid].mean()),
            "cost_total_usd": float(d["cost_usd"][valid].sum()),
            "cost_mean_usd": float(d["cost_usd"][valid].mean()),
            "U_acc_mean": float(d["U_acc"][valid].mean()),
            "N_infer_mean": float(d["N_infer"][valid].mean()),
            "battery_start_mwh": float(d["battery_mwh"][0]),
            "battery_end_mwh": float(d["battery_mwh_terminal"]),
            "battery_min_mwh": float(min(d["battery_mwh"].min(), d["battery_mwh_terminal"])),
            "battery_max_mwh": float(max(d["battery_mwh"].max(), d["battery_mwh_terminal"])),
            "charge_steps": int(d["charge"][valid].sum()),
            "charge_rate_pct": float(d["charge"][valid].mean() * 100),
            "E_charge_total_mwh": float(d["E_charge_mwh"][valid].sum()),
            "battery_source_pct": float((d["source"][valid] == 0).mean() * 100),
            "grid_source_pct": float((d["source"][valid] == 1).mean() * 100),
            "ci_mean": float(d["grid_ci"].mean()),
            "ci_min": float(d["grid_ci"].min()),
            "ci_max": float(d["grid_ci"].max()),
            "price_mean": float(d["grid_price"].mean()),
            "price_min": float(d["grid_price"].min()),
            "price_max": float(d["grid_price"].max()),
            "modes_used": {},
        }
        unique_modes, counts = np.unique(d["mode_index"], return_counts=True)
        for m, c in zip(unique_modes, counts):
            pair = env.mode_id_to_pair.get(int(m), None)
            if pair:
                n_id, h_id = pair
                name = f"{env.models[n_id].name}@hw{h_id}"
            else:
                name = "idle"
            ep["modes_used"][name] = int(c)
        ep_stats.append(ep)

    rewards = [e["reward_total"] for e in ep_stats]
    carbons = [e["carbon_total_g"] for e in ep_stats]
    costs = [e["cost_total_usd"] for e in ep_stats]

    if overlap > 0.0:
        tiling_desc = f"sliding window (overlap={overlap:.1%}, " f"stride={stride} slots)"
    else:
        tiling_desc = "sequential (non-overlapping, full split coverage)"

    summary = {
        "algorithm": algo_name,
        "split": split_name,
        "seed": seed,
        "n_episodes": n_ep,
        "episode_tiling": tiling_desc,
        "overlap": overlap,
        "stride_slots": stride,
        "steps_per_episode": int(len(all_data[0]["step"])),
        "cross_episode": {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "carbon_mean_g": float(np.mean(carbons)),
            "carbon_std_g": float(np.std(carbons)),
            "cost_mean_usd": float(np.mean(costs)),
            "cost_std_usd": float(np.std(costs)),
        },
        "episodes": ep_stats,
    }

    fname = f"{algo_name.lower()}_seed{seed}_summary.json"
    path = os.path.join(save_dir, fname)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved JSON: {path}")
    return summary


# Shared CLI parser; policies extend it with their own --args via extra_args_fn.
def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="./configs/exp_heavy.yaml")

    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test", "full"]
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Override output directory " "(default: results/<algo_name>)",
    )
    parser.add_argument(
        "--plot-first-n", type=int, default=2, help="Plot detailed trajectory for first N episodes"
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Episode overlap fraction in [0.0, 1.0). "
        "Overrides YAML episode.overlap. "
        "0.5 = 50%% sliding window overlap, "
        "0.0 = non-overlapping (default).",
    )
    return parser


# Main entry: parse CLI, build env, run all seeds × episodes, save outputs.
def evaluate_policy(
    policy_factory: Callable[[CarbonAwareEdgeEnv], BasePolicy],
    algo_name: str,
    description: str = "",
    extra_args_fn: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> tuple:

    parser = make_parser(description or f"{algo_name} evaluation")
    if extra_args_fn is not None:
        extra_args_fn(parser)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  {algo_name} — Evaluation")
    print("=" * 70)

    exp = ExperimentConfig(args.config)

    if args.save_dir:
        save_dir = args.save_dir
    else:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        save_dir = f"results/{exp.hardware_tier}/{config_name}/{algo_name.lower()}"
    os.makedirs(save_dir, exist_ok=True)

    if args.overlap is not None:
        overlap = args.overlap
    else:
        overlap = exp.episode_cfg.get("overlap", 0.0)
    exp.print_summary()

    split_info = exp.get_split(args.split)
    horizon_T = exp.episode_cfg["horizon_T"]
    delta_h = exp.episode_cfg.get("delta_hours", 0.25)

    ep_starts, stride, leftover = compute_sequential_ep_starts(
        split_info.n_slots,
        horizon_T,
        overlap=overlap,
    )
    n_episodes = len(ep_starts)

    if overlap > 0.0:
        tiling_label = f"sliding window, {overlap:.0%} overlap"
    else:
        tiling_label = "sequential, non-overlapping"

    print(f"\n{'='*70}")
    print(f"  EPISODE TILING ({args.split} split)")
    print(f"{'='*70}")
    print(f"  Split slots    : {split_info.n_slots:,} ({split_info.n_days:.0f} days)")
    print(f"  Horizon        : {horizon_T} slots ({horizon_T * delta_h / 24:.0f} days)")
    print(f"  Overlap        : {overlap:.1%}")
    print(f"  Stride         : {stride} slots ({stride * delta_h / 24:.1f} days)")
    print(f"  Episodes       : {n_episodes} ({tiling_label})")
    print(f"  Leftover slots : {leftover} ({leftover * delta_h / 24:.1f} days, discarded)")
    print(f"  Coverage       : {(ep_starts[-1] + horizon_T) / split_info.n_slots * 100:.1f}%")
    print(f"  Seeds          : {exp.seeds}")
    print(f"{'='*70}")

    seeds = exp.seeds
    all_seed_results: Dict[int, dict] = {}

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'#'*70}")
        print(f"  SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'#'*70}")

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = exp.make_env(args.split, seed=seed, verbose=(seed_idx == 0))

        if seed_idx == 0:
            print(f"\n  num_modes = {env.num_modes}")
            print(
                f"  action space = {env.num_modes} modes × 2 charge × 2 source = "
                f"{env.num_modes * 4}"
            )

            obs, info = env.reset(options={"ep_start": 0})
            mask = info["action_mask"]
            idle_any_valid = any(
                mask[0 * 4 + c * 2 + s]
                for c in range(2)
                for s in range(2)
                if 0 * 4 + c * 2 + s < len(mask)
            )
            print(f"\n  Idle mode actions valid? {idle_any_valid}")
            if idle_any_valid:
                print("  WARNING: idle is still QoS-admissible!")
            else:
                print("  Idle correctly excluded from QoS-admissible set")
            print(f"  Total valid actions: {mask.sum()} / {len(mask)}")

        policy = policy_factory(env)

        seed_dir = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        all_data: List[dict] = []
        all_episode_metrics: List[dict] = []

        for ep_idx, ep_start in enumerate(ep_starts):
            print(
                f"\n  Episode {ep_idx+1}/{n_episodes} "
                f"(slots {ep_start}–{ep_start + horizon_T}, "
                f"days {ep_start * delta_h / 24:.0f}–"
                f"{(ep_start + horizon_T) * delta_h / 24:.0f})..."
            )

            data = run_episode(env, policy, ep_start=ep_start)
            all_data.append(data)

            metrics = extract_episode_metrics(data, env)
            all_episode_metrics.append(metrics)

            print_episode_summary(data, env, algo_name, ep_idx=ep_idx, ep_start=ep_start)

            save_episode_csv(data, seed_dir, algo_name, episode_idx=ep_idx)

            if seed_idx == 0 and ep_idx < args.plot_first_n:
                plot_episode(data, env, seed_dir, algo_name, episode_idx=ep_idx, ep_start=ep_start)

        save_seed_summary_json(
            all_data,
            ep_starts,
            env,
            seed_dir,
            algo_name,
            args.split,
            seed,
            overlap=overlap,
            stride=stride,
        )

        if seed_idx == 0 and n_episodes > 1:
            plot_multi_episode_summary(
                all_data,
                seed_dir,
                algo_name,
                tiling_label=tiling_label,
            )

        all_seed_results[seed] = collect_seed_results(all_episode_metrics)

        rewards = [d["reward"].sum() for d in all_data]
        carbons = [d["carbon_g"][~d["invalid"]].sum() for d in all_data]
        costs = [d["cost_usd"][~d["invalid"]].sum() for d in all_data]
        n_invalid = [int(d["invalid"].sum()) for d in all_data]

        print(f"\n  --- Seed {seed}: {n_episodes} episodes ---")
        print(f"  Reward : {np.mean(rewards):+.4f} ± {np.std(rewards):.4f}")
        print(f"  Carbon : {np.mean(carbons):.6f} ± {np.std(carbons):.6f} gCO2")
        print(f"  Cost   : ${np.mean(costs):.8f} ± ${np.std(costs):.8f}")
        print(f"  Invalid: {np.mean(n_invalid):.1f} ± {np.std(n_invalid):.1f} per ep")

    print(f"\n{'='*70}")
    print(f"  CROSS-SEED AGGREGATION ({len(seeds)} seeds)")
    print(f"{'='*70}")

    agg = ExperimentConfig.aggregate_results(all_seed_results)
    ExperimentConfig.print_cross_seed_summary(agg, algo_name=algo_name, split_name=args.split)

    cross_seed_path = os.path.join(save_dir, f"{algo_name.lower()}_cross_seed.json")
    ExperimentConfig.save_results_json(agg, all_seed_results, cross_seed_path, algo_name=algo_name)

    print(f"\n{'='*70}")
    print(f"  {algo_name} — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Split          : {args.split}")
    print(f"  Split coverage : {split_info.n_slots:,} slots ({split_info.n_days:.0f} days)")
    print(f"  Tiling         : {tiling_label}")
    print(f"  Overlap        : {overlap:.1%}")
    print(f"  Stride         : {stride} slots ({stride * delta_h / 24:.1f} days)")
    print(f"  Episodes/seed  : {n_episodes}")
    print(f"  Steps/episode  : {horizon_T} ({horizon_T * delta_h / 24:.0f} days)")
    print(f"  Seeds          : {seeds}")
    print(f"  Leftover slots : {leftover} (discarded)")
    print(f"\n  Reward  : {agg['reward_mean']:+.4f} ± {agg['reward_std']:.4f}")
    print(f"  Carbon  : {agg['carbon_mean']:.6f} ± {agg['carbon_std']:.6f} gCO2")
    print(f"  Cost    : ${agg['cost_mean']:.8f} ± ${agg['cost_std']:.8f}")
    print(f"\n  Outputs saved to: {save_dir}/")
    print(f"    {algo_name.lower()}_cross_seed.json  (cross-seed aggregated)")
    for seed in seeds:
        print(f"    seed_{seed}/")
        print(f"      {algo_name.lower()}_seed{seed}_summary.json")
        print(f"      {algo_name.lower()}_ep{{0..{n_episodes-1}}}.csv")
    print(f"{'='*70}")

    return agg, all_seed_results, args
