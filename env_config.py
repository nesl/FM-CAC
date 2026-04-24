from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from env_simulator import (
    ModelVariant,
    HardwareState,
    ModeAttr,
    EnvConfig,
    CarbonAwareEdgeEnv,
    build_attr_table,
    compute_i_ref,
)

# Hardware tier -> YAML profile path.
_TIER_REGISTRY: Dict[str, dict] = {
    "heavy": {
        "yaml": "configs/heavy_hardware_config.yaml",
        "desc": "Jetson Orin Nano 8GB (YOLO detection)",
    },
}


# Parse the hardware profile YAML into model/hardware tables + latency/power maps.
def _load_hardware_config(
    config_path: str,
) -> Tuple[
    List[ModelVariant],
    List[HardwareState],
    List[str],
    Dict[Tuple[int, int], float],
    Optional[Dict[Tuple[int, int], float]],
]:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    models = [ModelVariant(name=m["name"], alpha=float(m["alpha"])) for m in data["models"]]

    hardwares = [
        HardwareState(
            unit=hw["unit"],
            f_clk=float(hw["f_clk"]),
            c_cores=int(hw["c_cores"]),
            p_cap=float(hw["p_cap"]),
        )
        for hw in data["hardwares"]
    ]

    hw_labels = [hw["label"] for hw in data["hardwares"]]
    measured_latencies = {
        (model_idx, hw_idx): float(latency)
        for model_idx, row in enumerate(data["latency_matrix"])
        for hw_idx, latency in enumerate(row)
    }

    measured_powers = None
    if "power_matrix" in data:
        measured_powers = {
            (model_idx, hw_idx): float(power_w)
            for model_idx, row in enumerate(data["power_matrix"])
            for hw_idx, power_w in enumerate(row)
        }

    return models, hardwares, hw_labels, measured_latencies, measured_powers


# Load (carbon-intensity, price) time series from a CSV; validates shape/signs.
def load_traces_from_csv(
    csv_path: str,
    ghg_col: str = "ghg_gCO2_per_kWh",
    price_col: str = "price_usd_per_kWh",
    delta_hours: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, parse_dates=[0])
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)

    g = df[ghg_col].values.astype(np.float32)
    p = df[price_col].values.astype(np.float32)

    min_slots = int(24 / delta_hours)
    assert len(g) == len(p), "GHG and price columns have different lengths"
    assert len(g) >= min_slots, (
        f"Trace too short: {len(g)} rows < {min_slots} for one day " f"at delta_hours={delta_hours}"
    )
    assert np.all(np.isfinite(g)), "GHG column contains NaN or Inf"
    assert np.all(np.isfinite(p)), "Price column contains NaN or Inf"
    assert np.all(g >= 0), (
        f"GHG column contains negative values (min={g.min():.4f}). "
        "Carbon intensity must be >= 0 gCO2/kWh."
    )

    print(f"[load_traces] {len(g)} rows from {csv_path}")
    print(f"  GHG  : mean={np.mean(g):.1f}  min={np.min(g):.1f}  max={np.max(g):.1f} gCO2/kWh")
    print(f"  Price: mean={np.mean(p):.4f}  min={np.min(p):.4f}  max={np.max(p):.4f} $/kWh")

    return g, p


# Synthesize a smooth diurnal CI + price trace for testing without real data.
def generate_synthetic_csv(
    csv_path: str,
    n_years: int = 5,
    seed: int = 42,
    delta_hours: float = 0.25,
) -> str:
    rng = np.random.default_rng(seed)
    slots_per_day = int(24 / delta_hours)
    days = 365 * n_years
    T = slots_per_day * days

    t = np.arange(T)
    hour_of_day = (t % slots_per_day) * delta_hours
    day_of_year = (t // slots_per_day) % 365

    daily_carbon = 80 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    seasonal_carbon = 40 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    noise_carbon = rng.normal(0, 15, T)
    g = (300 + daily_carbon + seasonal_carbon + noise_carbon).clip(50, 600)

    daily_price = 0.08 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
    afternoon_peak = 0.05 * np.exp(-0.5 * ((hour_of_day - 17) / 1.5) ** 2)
    seasonal_price = 0.03 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
    noise_price = rng.normal(0, 0.01, T)
    p = (0.15 + daily_price + afternoon_peak + seasonal_price + noise_price).clip(0.02, 0.50)

    start = pd.Timestamp("2020-01-01 00:00:00")
    freq_min = int(round(delta_hours * 60))
    timestamps = pd.date_range(start, periods=T, freq=f"{freq_min}min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ghg_gCO2_per_kWh": np.round(g, 1).astype(np.float32),
            "price_usd_per_kWh": np.round(p, 4).astype(np.float32),
        }
    )
    df.to_csv(csv_path, index=False)
    print(f"[generate] {T} rows ({days} days = {n_years} years) -> {csv_path}")
    return csv_path


# Max accuracy margin above the QoS floor, used to normalize the utility term.
def _compute_acc_ref(
    attr_table: Dict[Tuple[int, int], ModeAttr],
    u_acc: float,
    u_lat_s: float,
) -> float:
    admissible = [
        v for v in attr_table.values() if v.alpha >= u_acc and v.latency_s_per_infer <= u_lat_s
    ]
    if not admissible:
        return 1.0
    acc_ref = max(v.alpha - u_acc for v in admissible)
    return float(max(acc_ref, 1e-9))


# Assemble a CarbonAwareEdgeEnv from tier profiles + trace + battery/QoS knobs.
def _make_env(
    models: List[ModelVariant],
    hardwares: List[HardwareState],
    hw_labels: List[str],
    measured_latencies: Dict[Tuple[int, int], float],
    measured_powers: Optional[Dict[Tuple[int, int], float]],
    csv_path: Optional[str] = None,
    g: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
    horizon_T: int = 2880,
    delta_hours: float = 0.25,
    B_cap_mwh: float = 18_000.0,
    soc_min: float = 0.20,
    soc_max: float = 0.80,
    P_chg_mw: float = 20_000.0,
    eta_chg: float = 0.90,
    u_acc: float = 0.45,
    u_lat_s: float = 0.100,
    w_acc: float = 1.0,
    w_carb_infer: float = 1.0,
    w_carb_charge: float = 1.0,
    peukert_k: float = 1.05,
    V_nom: float = 3.7,
    infer_rate_ips: float = 1.0,
    camera_fps: float = 30.0,
    normalize_costs: bool = True,
    acc_ref: Optional[float] = None,
    g_ref_max: Optional[float] = None,
    invalid_action_penalty: float = -10.0,
    terminate_on_invalid: bool = False,
    disallow_charge_when_full: bool = True,
    use_coverage: bool = False,
    power_utilization: float = 0.8,
    seed: int = 0,
    verbose: bool = True,
) -> CarbonAwareEdgeEnv:

    if csv_path is not None:
        g, p = load_traces_from_csv(csv_path, delta_hours=delta_hours)
    elif g is None or p is None:
        raise ValueError("Provide either csv_path or both g and p arrays.")

    attr_table = build_attr_table(
        model_latencies=measured_latencies,
        models=models,
        hardwares=hardwares,
        power_utilization=power_utilization,
        model_powers=measured_powers,
    )

    if acc_ref is None:
        acc_ref = _compute_acc_ref(attr_table, u_acc, u_lat_s)
        if verbose:
            print(f"  Auto acc_ref = {acc_ref:.6f}  " f"-> U_acc = term_acc_n in [0, 1]")

    if verbose:
        print(f"\n{'='*90}")
        print(
            f"  Mode Table: {len(models)} models x {len(hardwares)} hardware = "
            f"{len(attr_table)} pairs"
        )
        print(f"{'='*90}")
        print(
            f"  {'#':>3} {'Model':<12} {'Hardware':<18} {'Alpha':>6} {'Lat(ms)':>8} "
            f"{'E/inf(mWh)':>11} {'Acc':>5} {'Lat':>5} {'Admit':>6}"
        )
        print(f"  {'-'*80}")

        n_admissible = 0
        row = 0
        for n_id, h_id in sorted(attr_table.keys()):
            attr = attr_table[(n_id, h_id)]
            acc_ok = "PASS" if attr.alpha >= u_acc else "FAIL"
            lat_ok = "PASS" if attr.latency_s_per_infer <= u_lat_s else "FAIL"
            admit = "YES" if (acc_ok == "PASS" and lat_ok == "PASS") else "no"
            if admit == "YES":
                n_admissible += 1
            print(
                f"  {row:>3} {models[n_id].name:<12} {hw_labels[h_id]:<18} "
                f"{attr.alpha:>6.3f} {attr.latency_s_per_infer*1000:>7.1f}ms "
                f"{attr.energy_mwh_per_infer:>10.4f} "
                f"{acc_ok:>5} {lat_ok:>5} {admit:>6}"
            )
            row += 1

        print(f"\n  Total: {len(attr_table)} + idle = {len(attr_table)+1}")
        print(f"  QoS-admissible: {n_admissible} (idle excluded)")

    admissible_items = [
        (k, v)
        for k, v in attr_table.items()
        if v.alpha >= u_acc and v.latency_s_per_infer <= u_lat_s
    ]
    if not admissible_items:
        raise ValueError(
            f"No admissible modes with u_acc={u_acc}, u_lat_s={u_lat_s}. "
            f"Lower the QoS thresholds."
        )

    T_s = delta_hours * 3600.0
    N_req = max(int(np.floor(infer_rate_ips * T_s + 1e-9)), 0)

    E_infer_max_mwh = 0.0
    for _k, _v in admissible_items:
        _lat = max(_v.latency_s_per_infer, 1e-9)
        _N_max = max(int(np.floor(T_s / _lat + 1e-9)), 0)
        _N = min(N_req, _N_max)
        _total = _N * _v.energy_mwh_per_infer
        if _total > E_infer_max_mwh:
            E_infer_max_mwh = _total
    E_infer_max_kwh = E_infer_max_mwh / 1_000_000.0

    E_charge_kwh = (P_chg_mw * delta_hours) / 1_000_000.0 / max(eta_chg, 1e-6)

    _g_max = float(g_ref_max) if g_ref_max is not None else float(np.max(g))
    carbon_ref_infer_g = max(_g_max * E_infer_max_kwh, 1e-6)
    carbon_ref_charge_g = max(_g_max * E_charge_kwh, 1e-6)

    cfg_for_iref = EnvConfig(
        delta_hours=delta_hours,
        infer_rate_ips=infer_rate_ips,
        u_acc=u_acc,
        u_lat_s=u_lat_s,
        V_nom=V_nom,
    )
    i_ref = compute_i_ref(attr_table, cfg_for_iref)

    if verbose:
        print(f"\n  Normalization refs (g_max x E_component):")
        print(
            f"    E_infer_max={E_infer_max_kwh*1e6:.2f} mWh  "
            f"E_charge={E_charge_kwh*1e6:.2f} mWh"
        )
        print(f"    carbon_ref_infer ={carbon_ref_infer_g:.6f} gCO2/slot")
        print(f"    carbon_ref_charge={carbon_ref_charge_g:.6f} gCO2/slot")
        print(f"  Peukert i_ref = {i_ref:.6f} A")

    cfg = EnvConfig(
        delta_hours=delta_hours,
        horizon_T=horizon_T,
        infer_rate_ips=infer_rate_ips,
        camera_fps=camera_fps,
        B_cap_mwh=B_cap_mwh,
        soc_min=soc_min,
        soc_max=soc_max,
        P_chg_mw=P_chg_mw,
        eta_chg=eta_chg,
        u_acc=u_acc,
        u_lat_s=u_lat_s,
        i_ref=i_ref,
        peukert_k=peukert_k,
        V_nom=V_nom,
        normalize_costs=normalize_costs,
        carbon_ref_infer_g=carbon_ref_infer_g,
        carbon_ref_charge_g=carbon_ref_charge_g,
        acc_ref=acc_ref,
        w_acc=w_acc,
        w_carb_infer=w_carb_infer,
        w_carb_charge=w_carb_charge,
        invalid_action_penalty=invalid_action_penalty,
        terminate_on_invalid=terminate_on_invalid,
        disallow_charge_when_full=disallow_charge_when_full,
        use_coverage=use_coverage,
    )

    env = CarbonAwareEdgeEnv(
        models=models,
        hardwares=hardwares,
        attr_table=attr_table,
        carbon_g_per_kwh=g,
        price_per_kwh=p,
        cfg=cfg,
        seed=seed,
    )

    if verbose:
        print(f"\n  Environment ready:")
        print(f"    Trace  : {len(g)} slots ({len(g)*delta_hours/24:.0f} days)")
        print(
            f"    Episode: {horizon_T} slots ({horizon_T*delta_hours/24:.0f} days / {horizon_T*delta_hours:.0f} hours)"
        )
        print(
            f"    Battery: {B_cap_mwh/1000:.0f} Wh, SoC=[{soc_min:.0%}, {soc_max:.0%}], "
            f"usable={env.B_usable_mwh/1000:.0f} Wh"
        )
        print(f"    Actions: {env.num_modes} modes x 2 charge x 2 source = " f"{env.num_modes * 4}")

    return env


# Lazy holder for a hardware tier: loaded profile + env factory.
class TierConfig:

    def __init__(self, tier: str, yaml_rel: str, desc: str):
        self.tier = tier
        self.desc = desc
        self._yaml_path = str(Path(__file__).parent / yaml_rel)

        (
            self.MODELS,
            self.HARDWARES,
            self.HW_LABELS,
            self.MEASURED_LATENCIES,
            self.MEASURED_POWERS,
        ) = _load_hardware_config(self._yaml_path)

    load_traces_from_csv = staticmethod(load_traces_from_csv)
    generate_synthetic_csv = staticmethod(generate_synthetic_csv)

    def make_env(self, **kwargs) -> CarbonAwareEdgeEnv:
        return _make_env(
            models=self.MODELS,
            hardwares=self.HARDWARES,
            hw_labels=self.HW_LABELS,
            measured_latencies=self.MEASURED_LATENCIES,
            measured_powers=self.MEASURED_POWERS,
            **kwargs,
        )


_TIER_CACHE: Dict[str, TierConfig] = {}


# Return the TierConfig for `tier`, loading + caching on first access.
def get_tier(tier: str) -> TierConfig:
    if tier not in _TIER_REGISTRY:
        raise ValueError(
            f"Unknown hardware_tier '{tier}'. " f"Choose from: {list(_TIER_REGISTRY.keys())}"
        )
    if tier not in _TIER_CACHE:
        info = _TIER_REGISTRY[tier]
        _TIER_CACHE[tier] = TierConfig(tier, info["yaml"], info["desc"])
    return _TIER_CACHE[tier]


def available_tiers() -> List[str]:
    return list(_TIER_REGISTRY.keys())
