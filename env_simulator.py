from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True)
class ModelVariant:
    name: str
    alpha: float


@dataclass(frozen=True)
class HardwareState:
    unit: str
    f_clk: float
    c_cores: int
    p_cap: float


@dataclass(frozen=True)
class ModeAttr:
    alpha: float
    latency_s_per_infer: float
    energy_mwh_per_infer: float


_BATTERY_EMPTY_THRESH_MWH: float = 1e-6


@dataclass(frozen=True)
class EnvConfig:

    delta_hours: float = 0.25
    horizon_T: int = 2880

    infer_rate_ips: float = 1.0
    camera_fps: float = 30.0

    B_cap_mwh: float = 18_000.0

    soc_min: float = 0.20
    soc_max: float = 0.80

    V_nom: float = 3.7
    i_ref: float = 0.065
    peukert_k: float = 1.05

    P_chg_mw: float = 20_000.0
    eta_chg: float = 0.90

    u_acc: float = 0.45
    u_lat_s: float = 0.100

    acc_ref: float = 1.0
    w_acc: float = 1.0
    w_carb_infer: float = 1.0
    w_carb_charge: float = 1.0

    normalize_costs: bool = True
    carbon_ref_infer_g: float = 1.0
    carbon_ref_charge_g: float = 1.0

    invalid_action_penalty: float = -10.0
    terminate_on_invalid: bool = False
    disallow_charge_when_full: bool = True
    use_coverage: bool = False


def build_attr_table(
    model_latencies: Dict[Tuple[int, int], float],
    models: List[ModelVariant],
    hardwares: List[HardwareState],
    power_utilization: float = 0.8,
    model_powers: Optional[Dict[Tuple[int, int], float]] = None,
) -> Dict[Tuple[int, int], ModeAttr]:
    attr_table: Dict[Tuple[int, int], ModeAttr] = {}
    for (n_id, h_id), latency_s in model_latencies.items():
        model = models[n_id]
        hardware = hardwares[h_id]
        if model_powers is not None and (n_id, h_id) in model_powers:
            power_mw = model_powers[(n_id, h_id)] * 1000.0
        else:
            power_mw = hardware.p_cap * 1000.0 * power_utilization
        latency_hours = latency_s / 3600.0
        energy_mwh = power_mw * latency_hours
        attr_table[(n_id, h_id)] = ModeAttr(
            alpha=model.alpha,
            latency_s_per_infer=latency_s,
            energy_mwh_per_infer=energy_mwh,
        )
    return attr_table


def compute_i_ref(
    attr_table: Dict[Tuple[int, int], ModeAttr],
    cfg_partial: EnvConfig,
) -> float:
    T_s = cfg_partial.delta_hours * 3600.0
    N_req = max(int(np.floor(cfg_partial.infer_rate_ips * T_s + 1e-9)), 0)

    min_E_raw_mwh = float("inf")
    for (_n_id, _h_id), attr in attr_table.items():
        if attr.alpha < cfg_partial.u_acc:
            continue
        if attr.latency_s_per_infer > cfg_partial.u_lat_s:
            continue

        lat = max(float(attr.latency_s_per_infer), 1e-9)
        N_max = max(int(np.floor(T_s / lat + 1e-9)), 0)

        N_infer = min(N_req, N_max)
        if N_infer <= 0:
            continue
        E_raw = float(N_infer) * float(attr.energy_mwh_per_infer)

        if E_raw < min_E_raw_mwh:
            min_E_raw_mwh = E_raw

    if min_E_raw_mwh == float("inf") or min_E_raw_mwh <= 0.0:
        return cfg_partial.i_ref

    P_dis_mw = min_E_raw_mwh / max(cfg_partial.delta_hours, 1e-12)
    i_ref = (P_dis_mw / 1000.0) / max(cfg_partial.V_nom, 1e-12)

    return float(i_ref)


class CarbonAwareEdgeEnv(gym.Env):

    metadata = {"render_modes": []}

    SRC_BATTERY = 0
    SRC_GRID = 1

    def __init__(
        self,
        models: List[ModelVariant],
        hardwares: List[HardwareState],
        attr_table: Dict[Tuple[int, int], ModeAttr],
        carbon_g_per_kwh: np.ndarray,
        price_per_kwh: Optional[np.ndarray] = None,
        cfg: Optional[EnvConfig] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)

        assert 0.0 <= self.cfg.soc_min < self.cfg.soc_max <= 1.0, (
            f"Need 0 <= soc_min < soc_max <= 1, got "
            f"soc_min={self.cfg.soc_min}, soc_max={self.cfg.soc_max}"
        )
        self.B_min_mwh: float = self.cfg.soc_min * self.cfg.B_cap_mwh
        self.B_max_mwh: float = self.cfg.soc_max * self.cfg.B_cap_mwh
        self.B_usable_mwh: float = self.B_max_mwh - self.B_min_mwh

        self.g_full = np.asarray(carbon_g_per_kwh, dtype=np.float32)
        if price_per_kwh is None:
            self.p_full = np.zeros_like(self.g_full, dtype=np.float32)
        else:
            self.p_full = np.asarray(price_per_kwh, dtype=np.float32)

        assert len(self.g_full) == len(
            self.p_full
        ), "carbon_g_per_kwh and price_per_kwh must have the same length"
        assert (
            len(self.g_full) >= self.cfg.horizon_T
        ), f"Trace length {len(self.g_full)} < horizon_T {self.cfg.horizon_T}"

        self.models = list(models)
        self.hardwares = list(hardwares)
        self.attr_table = dict(attr_table)

        self.N = len(self.models)
        self.H = len(self.hardwares)
        assert self.N > 0 and self.H > 0

        self.mode_id_to_pair: Dict[int, Optional[Tuple[int, int]]] = {0: None}
        self.pair_to_mode_id: Dict[Tuple[int, int], int] = {}
        next_id = 1
        for n_id, h_id in sorted(self.attr_table.keys()):
            self.mode_id_to_pair[next_id] = (n_id, h_id)
            self.pair_to_mode_id[(n_id, h_id)] = next_id
            next_id += 1
        self.num_modes = next_id

        _fmax = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=np.array(
                [self.B_min_mwh, 0.0, -_fmax, 0.0, -_fmax],
                dtype=np.float32,
            ),
            high=np.array(
                [self.B_max_mwh, _fmax, _fmax, _fmax, _fmax],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([self.num_modes, 2, 2])

        self._static_qos_mask: np.ndarray
        self._mode_E_out_mwh: np.ndarray
        self._mode_E_raw_mwh: np.ndarray
        self._precompute_static_mode_info()

        self.ep_start: int = 0
        self.t: int = 0
        self.b_mwh: float = 0.0
        self.b_embedded_carbon_g: float = 0.0
        self.b_embedded_cost_usd: float = 0.0

    def _precompute_static_mode_info(self) -> None:
        self._static_qos_mask = np.zeros(self.num_modes, dtype=bool)
        self._mode_E_out_mwh = np.zeros(self.num_modes, dtype=float)
        self._mode_E_raw_mwh = np.zeros(self.num_modes, dtype=float)

        for m in range(self.num_modes):
            if not self._qos_admissible(m):
                continue

            self._static_qos_mask[m] = True

            _, _, _, N_infer, _ = self._infer_counts(m)

            E_raw = self._E_infer_raw_mwh(m, N_infer)

            E_out = self._E_out_mwh_with_penalty(E_raw)
            self._mode_E_raw_mwh[m] = E_raw
            self._mode_E_out_mwh[m] = E_out

        assert self._static_qos_mask.any(), (
            f"No (model, hardware) mode satisfies QoS constraints "
            f"(u_acc={self.cfg.u_acc}, u_lat_s={self.cfg.u_lat_s}). "
            f"Action mask will be all-False."
        )

    @staticmethod
    def mwh_to_kwh(x_mwh: float) -> float:
        return x_mwh / 1_000_000.0

    def slot_seconds(self) -> float:
        return float(self.cfg.delta_hours * 3600.0)

    def _get_attr(self, mode_index: int) -> Optional[ModeAttr]:
        pair = self.mode_id_to_pair[mode_index]
        if pair is None:
            return None
        return self.attr_table[pair]

    def _qos_admissible(self, mode_index: int) -> bool:
        if mode_index == 0:
            return False

        attr = self._get_attr(mode_index)
        assert attr is not None
        return (attr.alpha >= self.cfg.u_acc) and (attr.latency_s_per_infer <= self.cfg.u_lat_s)

    def _infer_counts(self, mode_index: int) -> Tuple[int, int, int, int, float]:
        T = self.slot_seconds()
        N_frames = max(int(np.floor(self.cfg.camera_fps * T + 1e-9)), 0)

        N_req = max(int(np.floor(self.cfg.infer_rate_ips * T + 1e-9)), 0)

        if self.cfg.use_coverage and N_frames > 0:
            N_req = min(N_req, N_frames)

        if mode_index == 0:
            N_max = 0
            N_infer = 0
        else:
            attr = self._get_attr(mode_index)
            assert attr is not None
            lat = max(float(attr.latency_s_per_infer), 1e-9)
            N_max = max(int(np.floor(T / lat + 1e-9)), 0)
            N_infer = min(N_req, N_max)

        if self.cfg.use_coverage:
            coverage = min(float(N_infer / N_frames) if N_frames > 0 else 0.0, 1.0)
        else:
            coverage = 1.0 if (mode_index != 0 and N_infer > 0) else 0.0
        return N_frames, N_req, N_max, N_infer, float(coverage)

    def _E_in_mwh(self, charge_flag: int) -> float:
        return float(charge_flag) * self.cfg.P_chg_mw * self.cfg.delta_hours

    def _E_infer_raw_mwh(self, mode_index: int, N_infer: int) -> float:
        if mode_index == 0 or N_infer <= 0:
            return 0.0
        attr = self._get_attr(mode_index)
        assert attr is not None
        return float(N_infer) * float(attr.energy_mwh_per_infer)

    def _E_out_mwh_with_penalty(self, E_raw_mwh: float) -> float:
        if E_raw_mwh <= 0.0:
            return 0.0
        P_dis_mw = E_raw_mwh / max(self.cfg.delta_hours, 1e-12)

        i_t = (P_dis_mw / 1000.0) / max(self.cfg.V_nom, 1e-12)

        eta = max(1.0, (max(i_t, 1e-12) / max(self.cfg.i_ref, 1e-12)) ** (self.cfg.peukert_k - 1.0))
        return float(eta * E_raw_mwh)

    def _battery_effectively_at_floor(self) -> bool:
        if self.B_min_mwh > _BATTERY_EMPTY_THRESH_MWH:

            return False
        return self.b_mwh <= _BATTERY_EMPTY_THRESH_MWH

    def bat_avg_ci(self) -> float:
        if self._battery_effectively_at_floor():
            return 0.0
        return float(self.b_embedded_carbon_g / max(self.mwh_to_kwh(self.b_mwh), 1e-12))

    def bat_avg_price(self) -> float:
        if self._battery_effectively_at_floor():
            return 0.0
        return float(self.b_embedded_cost_usd / max(self.mwh_to_kwh(self.b_mwh), 1e-12))

    def _U_acc(self, mode_index: int, coverage: float) -> float:
        if mode_index == 0:
            return 0.0

        attr = self._get_attr(mode_index)
        assert attr is not None

        term_acc_n = max(float(attr.alpha) - self.cfg.u_acc, 0.0) / max(self.cfg.acc_ref, 1e-12)

        return float(coverage * term_acc_n)

    @staticmethod
    def _carbon_g(E_kwh: float, ci_g_per_kwh: float) -> float:
        return float(E_kwh) * float(ci_g_per_kwh)

    @staticmethod
    def _cost_usd(E_kwh: float, price_per_kwh: float) -> float:
        return float(E_kwh) * float(price_per_kwh)

    def _obs(self) -> np.ndarray:
        idx = min(self.ep_start + self.t, len(self.g_full) - 1)
        return np.array(
            [
                self.b_mwh,
                float(self.g_full[idx]),
                float(self.p_full[idx]),
                self.bat_avg_ci(),
                self.bat_avg_price(),
            ],
            dtype=np.float32,
        )

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.num_modes, 2, 2), dtype=bool)
        b_t = float(self.b_mwh)
        full = self.cfg.disallow_charge_when_full and (b_t >= self.B_max_mwh - 1e-9)

        for m in range(self.num_modes):
            if not self._static_qos_mask[m]:
                continue

            E_out = self._mode_E_out_mwh[m]

            for src in (self.SRC_BATTERY, self.SRC_GRID):

                if src == self.SRC_BATTERY and (b_t - E_out) < (self.B_min_mwh - 1e-9):
                    continue

                for c in (0, 1):
                    if full and c == 1:
                        continue

                    mask[m, c, src] = True

        return mask.reshape(-1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options and "ep_start" in options:
            self.ep_start = int(options["ep_start"])
        else:
            max_start = len(self.g_full) - self.cfg.horizon_T
            self.ep_start = int(self.rng.integers(0, max_start + 1))

        self.t = 0

        if options and "b_init_mwh" in options:
            self.b_mwh = float(options["b_init_mwh"])
        else:
            self.b_mwh = self.B_min_mwh

        self.b_mwh = float(np.clip(self.b_mwh, self.B_min_mwh, self.B_max_mwh))

        g0 = float(self.g_full[self.ep_start])
        p0 = float(self.p_full[self.ep_start])
        b_kwh = self.mwh_to_kwh(self.b_mwh)

        self.b_embedded_carbon_g = self._carbon_g(b_kwh, g0)
        self.b_embedded_cost_usd = self._cost_usd(b_kwh, p0)

        obs = self._obs()
        info = {"action_mask": self.get_action_mask(), "ep_start": self.ep_start}

        return obs, info

    def step(self, action):
        mode_index = int(action[0])
        charge_flag = int(action[1])
        source = int(action[2])

        if charge_flag not in (0, 1):
            raise ValueError(f"charge_flag must be 0 or 1, got {charge_flag}")
        if source not in (0, 1):
            raise ValueError(f"source must be 0 (battery) or 1 (grid), got {source}")
        if not (0 <= mode_index < self.num_modes):
            raise ValueError(f"mode_index {mode_index} out of range [0, {self.num_modes})")

        g_t = float(self.g_full[self.ep_start + self.t])
        p_t = float(self.p_full[self.ep_start + self.t])

        qos_ok = self._qos_admissible(mode_index)
        N_frames, N_req, N_max, N_infer, coverage = self._infer_counts(mode_index)

        E_raw_mwh = self._E_infer_raw_mwh(mode_index, N_infer)
        E_bat_mwh = self._E_out_mwh_with_penalty(E_raw_mwh)
        E_in_raw_mwh = self._E_in_mwh(charge_flag)

        feas_ok = True
        if source == self.SRC_BATTERY:

            feas_ok = (self.b_mwh - E_bat_mwh) >= (self.B_min_mwh - 1e-9)

        charge_ok = True
        if self.cfg.disallow_charge_when_full and charge_flag == 1:

            if self.b_mwh >= self.B_max_mwh - 1e-9:
                charge_ok = False

        src_ok = not (mode_index == 0 and source == self.SRC_BATTERY)

        invalid = not (qos_ok and feas_ok and charge_ok and src_ok)

        if invalid:
            self.t += 1
            terminated = bool(self.cfg.terminate_on_invalid)
            truncated = (not terminated) and (self.t >= self.cfg.horizon_T)
            return (
                self._obs(),
                float(self.cfg.invalid_action_penalty),
                terminated,
                truncated,
                {
                    "invalid_action": True,
                    "qos_ok": qos_ok,
                    "feas_ok": feas_ok,
                    "charge_ok": charge_ok,
                    "src_ok": src_ok,
                    "mode_index": mode_index,
                    "mode_pair": self.mode_id_to_pair[mode_index],
                    "charge": charge_flag,
                    "source": source,
                    "N_frames": N_frames,
                    "N_req": N_req,
                    "N_max": N_max,
                    "N_infer": N_infer,
                    "E_infer_bat_mwh": E_bat_mwh,
                    "E_infer_raw_mwh": E_raw_mwh,
                    "E_charge_accepted_mwh": 0.0,
                    "action_mask": (
                        self.get_action_mask() if not (terminated or truncated) else None
                    ),
                },
            )

        b_pre_discharge = float(self.b_mwh)

        carbon_grid_infer_g = 0.0
        cost_grid_infer_usd = 0.0
        carbon_batt_infer_g = 0.0
        cost_batt_infer_usd = 0.0

        if source == self.SRC_GRID:
            E_grid_kwh = self.mwh_to_kwh(E_raw_mwh)
            carbon_grid_infer_g = self._carbon_g(E_grid_kwh, g_t)
            cost_grid_infer_usd = self._cost_usd(E_grid_kwh, p_t)
        else:

            avg_ci = self.bat_avg_ci()
            avg_price = self.bat_avg_price()

            actual_drain_mwh = min(E_bat_mwh, self.b_mwh - self.B_min_mwh)
            actual_drain_kwh = self.mwh_to_kwh(actual_drain_mwh)
            carbon_batt_infer_g = self._carbon_g(actual_drain_kwh, avg_ci)
            cost_batt_infer_usd = self._cost_usd(actual_drain_kwh, avg_price)

            self.b_mwh = float(max(self.B_min_mwh, self.b_mwh - E_bat_mwh))
            self.b_embedded_carbon_g = float(
                max(0.0, self.b_embedded_carbon_g - carbon_batt_infer_g)
            )

            self.b_embedded_cost_usd = float(self.b_embedded_cost_usd - cost_batt_infer_usd)

            if self.b_mwh <= self.B_min_mwh + _BATTERY_EMPTY_THRESH_MWH:
                self.b_mwh = self.B_min_mwh

                if self.B_min_mwh <= _BATTERY_EMPTY_THRESH_MWH:
                    self.b_embedded_carbon_g = 0.0
                    self.b_embedded_cost_usd = 0.0

        carbon_grid_charge_g = 0.0
        cost_grid_charge_usd = 0.0
        accepted_charge_mwh = 0.0

        if charge_flag == 1 and E_in_raw_mwh > 0.0:
            headroom = max(0.0, self.B_max_mwh - b_pre_discharge)
            accepted_charge_mwh = min(E_in_raw_mwh, headroom)

            if accepted_charge_mwh > 0.0:
                acc_kwh = self.mwh_to_kwh(accepted_charge_mwh)

                grid_draw_kwh = acc_kwh / max(self.cfg.eta_chg, 1e-6)
                carbon_grid_charge_g = self._carbon_g(grid_draw_kwh, g_t)
                cost_grid_charge_usd = self._cost_usd(grid_draw_kwh, p_t)

                embedded_carbon_add = self._carbon_g(acc_kwh, g_t)
                embedded_cost_add = self._cost_usd(acc_kwh, p_t)
                self.b_embedded_carbon_g += embedded_carbon_add
                self.b_embedded_cost_usd += embedded_cost_add
                self.b_mwh = float(
                    min(
                        self.B_max_mwh,
                        self.b_mwh + accepted_charge_mwh,
                    )
                )

        U_acc = self._U_acc(mode_index, coverage)

        total_carbon_g = carbon_grid_infer_g + carbon_grid_charge_g
        total_cost_usd = cost_grid_infer_usd + cost_grid_charge_usd

        if self.cfg.normalize_costs:
            C_carb_infer = carbon_grid_infer_g / max(self.cfg.carbon_ref_infer_g, 1e-12)
            C_carb_charge = carbon_grid_charge_g / max(self.cfg.carbon_ref_charge_g, 1e-12)
        else:
            C_carb_infer = carbon_grid_infer_g
            C_carb_charge = carbon_grid_charge_g

        reward = float(
            self.cfg.w_acc * U_acc
            - self.cfg.w_carb_infer * C_carb_infer
            - self.cfg.w_carb_charge * C_carb_charge
        )

        self.t += 1
        terminated = False
        truncated = self.t >= self.cfg.horizon_T

        obs = self._obs()
        info = {
            "invalid_action": False,
            "mode_index": mode_index,
            "mode_pair": self.mode_id_to_pair[mode_index],
            "charge": charge_flag,
            "source": source,
            "b_mwh": self.b_mwh,
            "bat_avg_ci": self.bat_avg_ci(),
            "bat_avg_price": self.bat_avg_price(),
            "g_t": g_t,
            "p_t": p_t,
            "N_frames": N_frames,
            "N_req": N_req,
            "N_max": N_max,
            "N_infer": N_infer,
            "coverage": coverage,
            "E_infer_bat_mwh": E_bat_mwh,
            "E_infer_raw_mwh": E_raw_mwh,
            "E_charge_accepted_mwh": accepted_charge_mwh,
            "U_acc": U_acc,
            "total_carbon_g": total_carbon_g,
            "total_cost_usd": total_cost_usd,
            "C_carb_infer": C_carb_infer,
            "C_carb_charge": C_carb_charge,
            "action_mask": self.get_action_mask() if not truncated else None,
        }
        return obs, reward, terminated, truncated, info
