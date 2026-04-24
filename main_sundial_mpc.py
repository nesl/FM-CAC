#!/usr/bin/env python3
# TSFM-driven MPC: forecast -> DP -> action (mode, charge, source).

import time
import numpy as np
import argparse
import yaml
from collections import deque

from evaluator import BasePolicy, evaluate_policy


# Cold-start fallback: charge when CI below rolling percentile, else discharge.
class CarbonThresholdPolicy(BasePolicy):

    def __init__(self, env, ci_percentile: float = 40.0, window_size: int = 96 * 4):
        self.ci_percentile = ci_percentile
        self.window_size = window_size
        self.ci_history: deque = deque(maxlen=window_size)

        best_mode, best_alpha = None, -1.0
        for m in range(env.num_modes):
            pair = env.mode_id_to_pair.get(m)
            if pair is None:
                continue
            attr = env.attr_table[pair]
            if attr.alpha >= env.cfg.u_acc and attr.latency_s_per_infer <= env.cfg.u_lat_s:
                if attr.alpha > best_alpha:
                    best_alpha, best_mode = attr.alpha, m
        assert best_mode is not None, "No admissible mode found"
        self.fixed_mode = best_mode

    def reset_episode(self) -> None:
        self.ci_history.clear()

    def _is_clean(self, g_t: float) -> bool:
        self.ci_history.append(g_t)
        if len(self.ci_history) < 4:
            return g_t <= np.median(list(self.ci_history))
        return g_t <= np.percentile(list(self.ci_history), self.ci_percentile)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        g_t = float(obs[1])
        clean = self._is_clean(g_t)
        source = 1 if clean else 0
        charge = 1 if clean else 0
        mode = self.fixed_mode

        for c in (charge, 1 - charge):
            for s in (source, 1 - source):
                flat = mode * 4 + c * 2 + s
                if flat < len(mask) and mask[flat]:
                    return flat
        valid = np.where(mask)[0]
        return int(valid[0]) if len(valid) > 0 else 0


try:
    import torch
    from transformers import AutoModelForCausalLM

    SUNDIAL_AVAILABLE = True
except ImportError:
    SUNDIAL_AVAILABLE = False


# Zero-shot TSFM forecaster; falls back to persistence if load fails.
class SundialForecaster:
    def __init__(
        self, model_name="thuml/sundial-base-128m", n_samples=20, device="cuda", slots_per_hour=1
    ):
        self.n_samples = n_samples
        self.slots_per_hour = slots_per_hour
        self.device = device
        self.model = None

        if SUNDIAL_AVAILABLE:
            print(f"  [Sundial] Loading {model_name} ...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                self.model.eval()
                self.model.to(device)
                print(f"  [Sundial] Loaded on {device}")
            except Exception as e:
                print(f"  [Sundial] Load failed: {e}")
                self.model = None
        else:
            print("  [Sundial] torch/transformers not available. " "Using persistence fallback.")

    # Returns (mean, std) over n_samples.
    def forecast(self, context: np.ndarray, horizon: int = 24):
        if self.model is not None:
            return self._sundial_forecast(context, horizon)
        else:
            return self._persistence_fallback(context, horizon)

    def _sundial_forecast(self, context, horizon):
        seq = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        seq = seq.to(self.device)
        t0 = time.time()
        with torch.no_grad():
            output = self.model.generate(
                seq,
                max_new_tokens=horizon,
                num_samples=self.n_samples,
            )
        dt = time.time() - t0
        samples = output[0].cpu().numpy()
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        mean = np.maximum(mean, 0.0)
        hours = horizon / max(self.slots_per_hour, 1)
        print(
            f"    [Sundial] Forecast {hours:.0f}h ({horizon} slots, "
            f"{self.n_samples} samples) in {dt:.2f}s"
        )
        return mean, std

    def _persistence_fallback(self, context, horizon):
        week_slots = 168 * self.slots_per_hour
        period = min(week_slots, len(context))
        mean = np.zeros(horizon)
        for h in range(horizon):
            idx = len(context) - period + (h % period)
            idx = max(0, min(idx, len(context) - 1))
            mean[h] = context[idx]
        day_slots = max(24 * self.slots_per_hour, 1)
        base_std = np.std(context[-period:]) * 0.05 if period > 1 else 1.0
        std = np.array([base_std * (1.0 + 0.3 * np.sqrt(h / day_slots)) for h in range(horizon)])
        return mean, std


# Receding-horizon DP over a discretized battery grid. Deferred cost
# attribution charges battery discharge at a future-recharge percentile
# to prevent free-discharge past the look-ahead.
class MPCSolver:

    def __init__(
        self,
        env,
        gamma=0.998,
        n_battery_levels=100,
        carbon_scale=1.0,
        deferred_ci_percentile=25.0,
        deferred_weight=0.3,
        soc_guard_threshold=0.30,
        low_term_threshold=0.40,
        guard_penalty_mult=3.0,
    ):

        self.gamma = gamma
        self.n_levels = n_battery_levels
        self.carbon_scale = carbon_scale
        self.deferred_ci_percentile = deferred_ci_percentile
        self.deferred_weight = deferred_weight
        self.soc_guard_threshold = soc_guard_threshold
        self.low_term_threshold = low_term_threshold
        self.guard_penalty_mult = guard_penalty_mult

        self.B_min = env.B_min_mwh
        self.B_max = env.B_max_mwh
        self.B_levels = np.linspace(self.B_min, self.B_max, n_battery_levels)

        self.num_modes = env.num_modes
        self.delta_h = env.cfg.delta_hours
        delta_s = self.delta_h * 3600.0

        self.P_chg = env.cfg.P_chg_mw
        self.E_charge = self.P_chg * self.delta_h
        self.eta_chg = max(getattr(env.cfg, "eta_chg", 1.0), 1e-6)

        self.w_acc = env.cfg.w_acc
        self.w_carb_infer = env.cfg.w_carb_infer
        self.w_carb_charge = env.cfg.w_carb_charge

        self.normalize = getattr(env.cfg, "normalize_costs", True)
        self.carbon_ref_infer = max(getattr(env.cfg, "carbon_ref_infer_g", 1.0), 1e-12)
        self.carbon_ref_charge = max(getattr(env.cfg, "carbon_ref_charge_g", 1.0), 1e-12)
        self.acc_ref = max(env.cfg.acc_ref, 1e-12)

        self.V_nom = getattr(env.cfg, "V_nom", 3.7)
        self.peukert_k = getattr(env.cfg, "peukert_k", 1.05)
        self.i_ref = getattr(env.cfg, "i_ref", 1.0)

        ips = env.cfg.infer_rate_ips
        u_acc = env.cfg.u_acc
        u_lat_s = env.cfg.u_lat_s
        acc_ref = self.acc_ref

        self.mode_ids = []
        self.E_infer = {}
        self.U_acc = {}
        self.E_out_penalty = {}

        for mode_id, pair in env.mode_id_to_pair.items():
            if mode_id == 0:
                continue
            attr = env.attr_table[pair]

            if attr.alpha < u_acc or attr.latency_s_per_infer > u_lat_s:
                continue
            lat_s = attr.latency_s_per_infer
            e_per_infer = attr.energy_mwh_per_infer

            lat_s_safe = max(float(lat_s), 1e-9)
            N_max = max(int(np.floor(delta_s / lat_s_safe + 1e-9)), 0)
            N_req = max(int(np.floor(ips * delta_s + 1e-9)), 0)
            N_infer = min(N_req, N_max)
            E_raw = N_infer * e_per_infer
            self.E_infer[mode_id] = E_raw

            if E_raw > 0 and self.i_ref > 0:
                P_dis = E_raw / self.delta_h
                i_dis = (P_dis / 1000.0) / self.V_nom
                eta = max(1.0, (i_dis / self.i_ref) ** (self.peukert_k - 1))
            else:
                eta = 1.0
            self.E_out_penalty[mode_id] = eta * E_raw

            alpha = attr.alpha
            term_acc_n = max(0.0, float(alpha) - u_acc) / acc_ref
            self.U_acc[mode_id] = term_acc_n
            self.mode_ids.append(mode_id)

        self.actions = []
        for mode_id in self.mode_ids:
            for charge in [0, 1]:
                for source in [0, 1]:
                    flat = mode_id * 4 + charge * 2 + source
                    self.actions.append((flat, mode_id, charge, source))
        self.n_actions = len(self.actions)

        self._last_Q = None

        self._build_action_arrays()

        self._precompute_transitions()

        avg_E_infer = np.mean(list(self.E_infer.values())) if self.E_infer else 1.0

        self._charge_infer_ratio = self.E_charge / max(avg_E_infer, 1e-6)

        print(
            f"  [MPCSolver] {len(self.mode_ids)} modes × 4 = "
            f"{self.n_actions} actions, {n_battery_levels} battery levels"
        )
        print(
            f"  [MPCSolver] E_charge={self.E_charge:.1f} mWh/slot, "
            f"avg E_infer={avg_E_infer:.1f} mWh/slot, "
            f"ratio={self._charge_infer_ratio:.1f}×"
        )
        print(f"  [MPCSolver] B_range=[{self.B_min:.0f}, {self.B_max:.0f}] mWh")
        print(f"  [MPCSolver] carbon_scale={self.carbon_scale:.1f}")
        print(
            f"  [MPCSolver] deferred_ci_percentile={self.deferred_ci_percentile:.0f}th, "
            f"deferred_weight={self.deferred_weight:.2f} "
            f"(battery discharge cost = weight × pct_CI × E_out)"
        )
        print(
            f"  [MPCSolver] Deferred battery carbon in "
            f"weighted-sum — discharge is costed at forecast CI percentile"
        )

        for m in sorted(self.mode_ids):
            pair = env.mode_id_to_pair.get(m)
            name = f"{env.models[pair[0]].name}/hw{pair[1]}" if pair else "?"
            print(
                f"    mode {m}: {name:25s}  E_infer={self.E_infer[m]:8.1f} mWh  "
                f"E_out={self.E_out_penalty[m]:8.1f} mWh  "
                f"U_acc={self.U_acc[m]:.4f}"
            )

    def _build_action_arrays(self):

        self.act_flat = np.array([a[0] for a in self.actions], dtype=np.int32)

        self.act_mode = np.array([a[1] for a in self.actions], dtype=np.int32)

        self.act_charge = np.array([a[2] for a in self.actions], dtype=np.int32)

        self.act_source = np.array([a[3] for a in self.actions], dtype=np.int32)

        self.act_E_infer = np.array([self.E_infer[a[1]] for a in self.actions], dtype=np.float64)

        self.act_E_out = np.array(
            [self.E_out_penalty[a[1]] for a in self.actions], dtype=np.float64
        )

        self.act_U_acc = np.array([self.U_acc[a[1]] for a in self.actions], dtype=np.float64)

        self.act_is_battery = (self.act_source == 0).astype(np.float64)
        self.act_is_grid = (self.act_source == 1).astype(np.float64)
        self.act_is_charging = self.act_charge.astype(np.float64)

    def _precompute_transitions(self):

        L = self.n_levels

        b_L1 = self.B_levels[:, None]
        E_out_1A = self.act_E_out[None, :]
        bat_1A = self.act_is_battery[None, :]
        chg_1A = self.act_is_charging[None, :]

        b_after = b_L1 - bat_1A * E_out_1A

        self._trans_feasible = b_after >= self.B_min - 1e-6

        full_mask = b_L1 >= self.B_max - 1e-6
        self._trans_feasible &= ~(full_mask & (chg_1A > 0.5))

        headroom = np.maximum(self.B_max - b_L1, 0.0)
        E_in = chg_1A * np.minimum(self.E_charge, headroom)
        self._trans_E_in = E_in

        b_next = np.clip(b_after + E_in, self.B_min, self.B_max)

        self._trans_next_idx = self._snap_idx_vec(b_next.ravel()).reshape(L, self.n_actions)

    def _snap_idx(self, b):

        idx = int(np.searchsorted(self.B_levels, b))

        idx = min(max(idx, 0), self.n_levels - 1)

        if idx > 0 and abs(self.B_levels[idx - 1] - b) < abs(self.B_levels[idx] - b):
            idx -= 1
        return idx

    def _snap_idx_vec(self, b_arr):

        idx = np.searchsorted(self.B_levels, b_arr)

        idx = np.clip(idx, 0, self.n_levels - 1)

        prev_idx = np.maximum(idx - 1, 0)
        use_prev = np.abs(self.B_levels[prev_idx] - b_arr) < np.abs(self.B_levels[idx] - b_arr)
        use_prev &= idx > 0

        idx[use_prev] = prev_idx[use_prev]
        return idx

    def apply_green_mode(self, green_grid_mode, green_battery_mode):
        for i, (_flat, _mode_id, _charge, source) in enumerate(self.actions):

            actual_mode = green_grid_mode if source == 1 else green_battery_mode

            self.act_E_infer[i] = self.E_infer[actual_mode]
            self.act_E_out[i] = self.E_out_penalty[actual_mode]
            self.act_U_acc[i] = self.U_acc[actual_mode]

        self._precompute_transitions()

    # Backward induction over (battery × horizon); returns (L, H) action table.
    def solve_full(self, g_forecast, conf):
        H = len(g_forecast)
        if H == 0:
            return np.zeros((self.n_levels, 1), dtype=np.int32)

        L = self.n_levels
        A = self.n_actions
        policy = np.zeros((L, H), dtype=np.int32)
        gamma_step = self.gamma

        pct = self.deferred_ci_percentile
        g_deferred = np.zeros(H)
        for k in range(H):
            future_slice = g_forecast[k:]
            g_deferred[k] = np.percentile(future_slice, pct)

        g_high = np.percentile(g_forecast, 75)
        if self.normalize:
            save_per_mwh = self.w_carb_charge * (g_high / 1e6) / self.carbon_ref_charge
        else:
            save_per_mwh = self.w_carb_charge * g_high / 1e6

        avg_tail_conf = np.mean(conf[int(0.75 * H) :]) if H > 4 else 1.0
        terminal_scale = 0.5 + 0.5 * avg_tail_conf

        soc_frac_term = (self.B_levels - self.B_min) / max(self.B_max - self.B_min, 1.0)
        term_deficit = (
            np.maximum(self.low_term_threshold - soc_frac_term, 0.0) / self.low_term_threshold
        )
        V_terminal = -terminal_scale * save_per_mwh * (self.B_max - self.B_min) * term_deficit**2
        V_next = V_terminal.copy()

        trans_next = self._trans_next_idx
        trans_ok = self._trans_feasible
        trans_E_in = self._trans_E_in

        soc_frac = (self.B_levels - self.B_min) / max(self.B_max - self.B_min, 1.0)
        deficit = np.maximum(self.soc_guard_threshold - soc_frac, 0.0) / self.soc_guard_threshold
        guard_scale = self.w_acc * self.guard_penalty_mult
        b_shape = -guard_scale * deficit**2

        E_infer_A = self.act_E_infer
        E_out_A = self.act_E_out
        is_grid_A = self.act_is_grid
        is_bat_A = self.act_is_battery

        cs = self.carbon_scale

        for k in range(H - 1, -1, -1):
            g_k = g_forecast[k]
            g_future_k = g_deferred[k]

            C_grid_infer = is_grid_A * g_k * E_infer_A / 1e6

            C_charge = g_k * trans_E_in / (self.eta_chg * 1e6)

            C_bat_deferred = self.deferred_weight * is_bat_A * g_future_k * E_out_A / 1e6

            C_infer_total = C_grid_infer[None, :] + C_bat_deferred[None, :]

            # DP objective: accuracy margin minus carbon (infer + charge).
            # Latency is enforced as a hard QoS constraint upstream (the mask
            # pre-filters modes with l > u_lat), not rewarded here. Electricity
            # cost is logged in the per-step CSV but not in this objective.
            if self.normalize:
                R = (
                    self.w_acc * self.act_U_acc[None, :]
                    - cs * self.w_carb_infer * C_infer_total / self.carbon_ref_infer
                    - cs * self.w_carb_charge * C_charge / self.carbon_ref_charge
                )
            else:
                R = (
                    self.w_acc * self.act_U_acc[None, :]
                    - cs * self.w_carb_infer * C_infer_total
                    - cs * self.w_carb_charge * C_charge
                )

            V_future = V_next[trans_next]
            Q = R + gamma_step * V_future
            Q += b_shape[trans_next]
            Q[~trans_ok] = -np.inf

            best_a = np.argmax(Q, axis=1)
            V_current = Q[np.arange(L), best_a]
            all_inf = np.all(Q == -np.inf, axis=1)
            V_current[all_inf] = 0.0

            policy[:, k] = best_a
            V_next = V_current

            if k == 0:
                self._last_Q = Q.copy()

        return policy

    def lookup(self, policy, k, b_current, prev_mode, mask):
        H = policy.shape[1]

        if k < 0 or k >= H:
            valid = np.where(mask)[0]
            return int(valid[0]) if len(valid) > 0 else 0

        bi = self._snap_idx(b_current)

        best_ai = policy[bi, k]
        if 0 <= best_ai < self.n_actions:
            flat = int(self.act_flat[best_ai])
            if flat < len(mask) and mask[flat]:
                return flat

        q_row = self._last_Q[bi] if self._last_Q is not None else None
        if q_row is not None:
            for ai in np.argsort(-q_row):
                flat = int(self.act_flat[ai])
                if flat < len(mask) and mask[flat]:
                    return flat

        valid = np.where(mask)[0]
        return int(valid[0]) if len(valid) > 0 else 0


# Main control loop: cold-start -> forecast every K -> DP slice -> reactive override.
class SundialMPCPolicy(BasePolicy):

    def __init__(
        self,
        env,
        gamma=0.998,
        n_battery_levels=100,
        n_samples=20,
        k_reforecast=96,
        forecast_horizon=96,
        context_length=1344,
        cold_start_steps=96,
        cold_start_ci_percentile=40.0,
        sundial_model="thuml/sundial-base-128m",
        device="cuda",
        reactive_charge_pct=0.20,
        reactive_discharge_pct=0.75,
        green_mode=False,
        carbon_scale=5.0,
        deferred_ci_percentile=25.0,
        deferred_weight=0.3,
        soc_guard_threshold=0.30,
        low_term_threshold=0.40,
        guard_penalty_mult=3.0,
    ):
        self.env = env

        self.gamma = gamma
        self.n_samples = n_samples
        self.k_reforecast = k_reforecast
        self.H_slots = forecast_horizon
        self.forecast_slots = 2 * self.H_slots
        assert k_reforecast <= self.forecast_slots, (
            f"k_reforecast ({k_reforecast}) must be <= forecast_slots "
            f"({self.forecast_slots}), otherwise the forecast is exhausted "
            f"before the next reforecast and the DP degrades to a myopic policy."
        )
        self.context_length_slots = context_length
        self.delta_h = env.cfg.delta_hours
        self.slots_per_hour = max(1, int(round(1.0 / self.delta_h)))

        self.reactive_charge_pct = reactive_charge_pct
        self.reactive_discharge_pct = reactive_discharge_pct
        self.green_mode = green_mode

        self.cold_start_steps = cold_start_steps
        self.fallback = CarbonThresholdPolicy(env, ci_percentile=cold_start_ci_percentile)

        self.forecaster = SundialForecaster(
            model_name=sundial_model,
            n_samples=n_samples,
            device=device,
            slots_per_hour=self.slots_per_hour,
        )

        self.solver = MPCSolver(
            env=env,
            gamma=gamma,
            n_battery_levels=n_battery_levels,
            carbon_scale=carbon_scale,
            deferred_ci_percentile=deferred_ci_percentile,
            deferred_weight=deferred_weight,
            soc_guard_threshold=soc_guard_threshold,
            low_term_threshold=low_term_threshold,
            guard_penalty_mult=guard_penalty_mult,
        )

        self._build_action_lookup()
        self._reset_state()

        if green_mode:

            admissible = [m for m in self.solver.mode_ids if self.solver.U_acc[m] > 0]

            self.green_grid_mode = min(admissible, key=lambda m: self.solver.E_infer[m])

            self.green_battery_mode = max(admissible, key=lambda m: self.solver.U_acc[m])

            g_mode_pair = env.mode_id_to_pair.get(self.green_grid_mode)
            b_mode_pair = env.mode_id_to_pair.get(self.green_battery_mode)
            g_name = f"{env.models[g_mode_pair[0]].name}/hw{g_mode_pair[1]}" if g_mode_pair else "?"
            b_name = f"{env.models[b_mode_pair[0]].name}/hw{b_mode_pair[1]}" if b_mode_pair else "?"
            print(f"  [Policy] GREEN MODE enabled:")
            print(f"    Grid mode   : id={self.green_grid_mode} ({g_name})")
            print(f"    Battery mode: id={self.green_battery_mode} ({b_name})")

            self.solver.apply_green_mode(self.green_grid_mode, self.green_battery_mode)

        print(
            f"  [Policy] carbon_scale={carbon_scale:.1f}, "
            f"deferred_ci_pct={deferred_ci_percentile:.0f}th, "
            f"deferred_weight={deferred_weight:.2f}"
        )
        print(
            f"  [Policy] Cold-start: "
            f"{'green low-energy' if green_mode else 'CarbonThreshold'} "
            f"for first {self.cold_start_steps} steps"
        )
        print(
            f"  [Policy] TSFM+MPC from step {self.cold_start_steps}: "
            f"look-ahead={self.H_slots} slots ({self.H_slots/self.slots_per_hour:.0f}h), "
            f"TSFM forecast={self.forecast_slots} slots ({self.forecast_slots/self.slots_per_hour:.0f}h), "
            f"max context={self.context_length_slots} slots ({self.context_length_slots/self.slots_per_hour:.0f}h)"
        )
        print(
            f"  [Policy] Reforecast every {k_reforecast} slots ({k_reforecast/self.slots_per_hour:.0f}h)"
        )
        print(
            f"  [Policy] Reactive thresholds: "
            f"charge<{reactive_charge_pct:.0%}, battery>{reactive_discharge_pct:.0%}"
        )

    def _build_action_lookup(self):

        self._action_map = {}
        for flat_id, mode_id, charge, source in self.solver.actions:
            self._action_map[(mode_id, charge, source)] = flat_id

        self._default_mode = self.solver.mode_ids[0] if self.solver.mode_ids else 1

    def _reset_state(self):
        self.ci_history = []

        self._sundial_ci_mean = None
        self._sundial_ci_std = None
        self._sundial_step = 0

        self.prev_mode = 0
        self.step_count = 0

        self.ci_ema = None
        self.ci_ema_alpha = 0.005
        self.ci_running_min = None
        self.ci_running_max = None
        self.ci_buffer = []
        self.ci_calibrated = False

    def reset_episode(self):
        self._reset_state()
        self.fallback.reset_episode()

    def _compute_confidence(self, ci_std, ci_mean):
        N = len(ci_mean)
        eps = 1e-6

        cv_ci = ci_std / (np.abs(ci_mean) + eps)

        conf_B = 1.0 / (1.0 + cv_ci)

        slot_positions = np.arange(N)
        conf_A = self.gamma**slot_positions
        return conf_A * conf_B

    def _update_ci_stats(self, g_t):
        if not self.ci_calibrated:

            self.ci_buffer.append(g_t)
            if len(self.ci_buffer) >= 96:
                self.ci_ema = np.mean(self.ci_buffer)
                self.ci_running_min = np.percentile(self.ci_buffer, 10)
                self.ci_running_max = np.percentile(self.ci_buffer, 90)
                self.ci_calibrated = True
                self.ci_buffer = None
            return

        alpha = self.ci_ema_alpha
        self.ci_ema = alpha * g_t + (1.0 - alpha) * self.ci_ema

        decay = 0.001
        self.ci_running_min = min(
            g_t, self.ci_running_min + decay * (self.ci_ema - self.ci_running_min)
        )
        self.ci_running_max = max(
            g_t, self.ci_running_max - decay * (self.ci_running_max - self.ci_ema)
        )

    # Reforecast every K slots; 2H horizon sliced per-step into DP's H window.
    def _generate_forecast(self):

        ci_context = np.array(self.ci_history)
        L = self.context_length_slots
        if len(ci_context) > L:
            ci_context = ci_context[-L:]

        F = self.forecast_slots
        ci_mean, ci_std = self.forecaster.forecast(ci_context, horizon=F)

        self._sundial_ci_mean = ci_mean
        self._sundial_ci_std = ci_std
        self._sundial_step = self.step_count

    def _build_current_forecast(self, g_t):
        elapsed = self.step_count - self._sundial_step
        remaining_mean = self._sundial_ci_mean[elapsed:]
        remaining_std = self._sundial_ci_std[elapsed:]

        g_forecast = np.concatenate([[g_t], remaining_mean])
        s_forecast = np.concatenate([[0.0], remaining_std])

        conf = self._compute_confidence(s_forecast, g_forecast)
        conf[0] = 1.0

        return g_forecast, conf

    # SoC safety net: force-charge when low+clean, force-discharge when high+dirty.
    def _reactive_override(self, dp_action, g_t, p_t, b_t, mask):

        if not self.ci_calibrated:
            return dp_action, False

        ci_range = max(self.ci_running_max - self.ci_running_min, 1.0)
        ci_position = (g_t - self.ci_running_min) / ci_range

        dp_mode = (dp_action // 2) // 2
        if dp_mode == 0:
            dp_mode = self._default_mode

        if self.green_mode:
            charge_mode = self.green_grid_mode
            discharge_mode = self.green_battery_mode
        else:
            charge_mode = dp_mode
            discharge_mode = dp_mode

        B_headroom = self.solver.B_max - b_t
        B_usable = b_t - self.solver.B_min

        min_headroom = max(self.solver.E_charge * 0.05, 100.0)
        if ci_position < self.reactive_charge_pct and B_headroom > min_headroom:
            desired = self._action_map.get((charge_mode, 1, 1), None)
            if desired is not None and desired < len(mask) and mask[desired]:
                return desired, True

        if ci_position > self.reactive_discharge_pct and B_usable > 0:
            mode_E_out = self.solver.E_out_penalty.get(discharge_mode, 0)
            if B_usable >= mode_E_out:
                desired = self._action_map.get((discharge_mode, 0, 0), None)
                if desired is not None and desired < len(mask) and mask[desired]:
                    return desired, True

        return dp_action, False

    def _green_cold_start(self, obs, mask):
        mode = self.green_grid_mode
        source = 1
        for charge in [1, 0]:
            flat = mode * 4 + charge * 2 + source
            if flat < len(mask) and mask[flat]:
                return flat

        valid = np.where(mask)[0]
        return int(valid[0]) if len(valid) > 0 else 0

    def _apply_green_mode(self, action, mask):

        source = action % 2
        charge = (action // 2) % 2

        mode = self.green_grid_mode if source == 1 else self.green_battery_mode

        green_flat = mode * 4 + charge * 2 + source
        if green_flat < len(mask) and mask[green_flat]:
            return green_flat

        fallback_flat = mode * 4 + (1 - charge) * 2 + source
        if fallback_flat < len(mask) and mask[fallback_flat]:
            return fallback_flat

        grid_flat = self.green_grid_mode * 4 + 0 * 2 + 1
        if grid_flat < len(mask) and mask[grid_flat]:
            return grid_flat

        return action

    # obs = [b_t, g_t, p_t, ...].
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        b_t = obs[0]
        g_t = obs[1]
        p_t = obs[2]

        self.ci_history.append(g_t)
        self._update_ci_stats(g_t)

        if self.step_count < self.cold_start_steps:
            self.step_count += 1
            if self.green_mode:
                return self._green_cold_start(obs, mask)
            return self.fallback.act(obs, mask)

        need_sundial = (
            self._sundial_ci_mean is None
            or (self.step_count - self._sundial_step) >= self.k_reforecast
        )
        if need_sundial:
            self._generate_forecast()

        g_forecast, conf = self._build_current_forecast(g_t)

        policy = self.solver.solve_full(g_forecast, conf)

        dp_action = self.solver.lookup(
            policy=policy,
            k=0,
            b_current=b_t,
            prev_mode=self.prev_mode,
            mask=mask,
        )

        action, _ = self._reactive_override(dp_action, g_t, p_t, b_t, mask)

        if self.green_mode:
            action = self._apply_green_mode(action, mask)

        source = action % 2
        rem = action // 2
        charge = rem % 2
        mode = rem // 2
        self.prev_mode = mode

        self.step_count += 1

        return action


if __name__ == "__main__":

    _HARDCODED_DEFAULTS = dict(
        gamma=0.998,
        n_samples=20,
        battery_levels=100,
        k_reforecast=96,
        forecast_horizon=96,
        context_length=1344,
        cold_start_steps=96,
        cold_start_ci_percentile=40.0,
        sundial_model="thuml/sundial-base-128m",
        device="cuda",
        reactive_charge_pct=0.20,
        reactive_discharge_pct=0.75,
        green_mode=False,
        carbon_scale=5.0,
        deferred_ci_percentile=25.0,
        deferred_weight=0.3,
        soc_guard_threshold=0.30,
        low_term_threshold=0.40,
        guard_penalty_mult=3.0,
    )

    _pre_config = argparse.ArgumentParser(add_help=False)
    _pre_config.add_argument("--config", type=str, default="configs/exp_heavy.yaml")
    _pre_known, _ = _pre_config.parse_known_args()

    _DEFAULTS = dict(_HARDCODED_DEFAULTS)
    try:
        with open(_pre_known.config, "r") as _f:
            _yaml_cfg = yaml.safe_load(_f)
        _mpc_yaml = _yaml_cfg.get("mpc_policy", {})

        _YAML_TO_KEY = {
            "gamma": "gamma",
            "n_samples": "n_samples",
            "n_battery_levels": "battery_levels",
            "k_reforecast": "k_reforecast",
            "forecast_horizon": "forecast_horizon",
            "context_length": "context_length",
            "cold_start_steps": "cold_start_steps",
            "cold_start_ci_percentile": "cold_start_ci_percentile",
            "sundial_model": "sundial_model",
            "device": "device",
            "reactive_charge_pct": "reactive_charge_pct",
            "reactive_discharge_pct": "reactive_discharge_pct",
            "green_mode": "green_mode",
            "carbon_scale": "carbon_scale",
            "deferred_ci_percentile": "deferred_ci_percentile",
            "deferred_weight": "deferred_weight",
            "soc_guard_threshold": "soc_guard_threshold",
            "low_term_threshold": "low_term_threshold",
            "guard_penalty_mult": "guard_penalty_mult",
        }
        for yaml_key, default_key in _YAML_TO_KEY.items():
            if yaml_key in _mpc_yaml:
                _DEFAULTS[default_key] = _mpc_yaml[yaml_key]
    except (FileNotFoundError, KeyError):
        pass

    def _add_mpc_args(parser):
        parser.add_argument("--gamma", type=float, default=_DEFAULTS["gamma"])
        parser.add_argument("--n-samples", type=int, default=_DEFAULTS["n_samples"])
        parser.add_argument("--battery-levels", type=int, default=_DEFAULTS["battery_levels"])
        parser.add_argument("--k-reforecast", type=int, default=_DEFAULTS["k_reforecast"])
        parser.add_argument("--forecast-horizon", type=int, default=_DEFAULTS["forecast_horizon"])
        parser.add_argument("--context-length", type=int, default=_DEFAULTS["context_length"])
        parser.add_argument(
            "--cold-start-steps",
            type=int,
            default=_DEFAULTS["cold_start_steps"],
            help="Slots of heuristic fallback before MPC starts (96 = 24h)",
        )
        parser.add_argument(
            "--cold-start-ci-percentile",
            type=float,
            default=_DEFAULTS["cold_start_ci_percentile"],
            help="CI percentile threshold for CarbonThreshold fallback during cold-start",
        )
        parser.add_argument("--sundial-model", type=str, default=_DEFAULTS["sundial_model"])
        parser.add_argument("--device", type=str, default=_DEFAULTS["device"])
        parser.add_argument(
            "--reactive-charge-pct", type=float, default=_DEFAULTS["reactive_charge_pct"]
        )
        parser.add_argument(
            "--reactive-discharge-pct", type=float, default=_DEFAULTS["reactive_discharge_pct"]
        )
        parser.add_argument(
            "--carbon-scale",
            type=float,
            default=_DEFAULTS["carbon_scale"],
            help="DP carbon penalty multiplier (higher = more carbon reduction)",
        )
        parser.add_argument(
            "--deferred-ci-percentile",
            type=float,
            default=_DEFAULTS["deferred_ci_percentile"],
            help="Percentile of forecast CI used for deferred battery cost "
            "(lower = more battery use, 25=default, 50=mean-like)",
        )
        parser.add_argument(
            "--deferred-weight",
            type=float,
            default=_DEFAULTS["deferred_weight"],
            help="Scale factor for deferred battery cost to avoid double-counting "
            "with DP state transitions (0.0=free discharge, 1.0=full cost)",
        )
        parser.add_argument(
            "--soc-guard-threshold",
            type=float,
            default=_DEFAULTS["soc_guard_threshold"],
            help="SoC fraction (of usable range) below which guard penalty kicks in",
        )
        parser.add_argument(
            "--low-term-threshold",
            type=float,
            default=_DEFAULTS["low_term_threshold"],
            help="Terminal SoC fraction below which end-of-horizon penalty applies",
        )
        parser.add_argument(
            "--guard-penalty-mult",
            type=float,
            default=_DEFAULTS["guard_penalty_mult"],
            help="Multiplier on accuracy weight for SoC guard penalty strength",
        )
        grp = parser.add_mutually_exclusive_group()
        grp.add_argument(
            "--green-mode", dest="green_mode", action="store_true", default=_DEFAULTS["green_mode"]
        )
        grp.add_argument("--no-green-mode", dest="green_mode", action="store_false")

    pre = argparse.ArgumentParser(add_help=False)
    _add_mpc_args(pre)
    known, _ = pre.parse_known_args()

    evaluate_policy(
        policy_factory=lambda env: SundialMPCPolicy(
            env=env,
            gamma=known.gamma,
            n_battery_levels=known.battery_levels,
            n_samples=known.n_samples,
            k_reforecast=known.k_reforecast,
            forecast_horizon=known.forecast_horizon,
            context_length=known.context_length,
            cold_start_steps=known.cold_start_steps,
            cold_start_ci_percentile=known.cold_start_ci_percentile,
            sundial_model=known.sundial_model,
            device=known.device,
            reactive_charge_pct=known.reactive_charge_pct,
            reactive_discharge_pct=known.reactive_discharge_pct,
            green_mode=known.green_mode,
            carbon_scale=known.carbon_scale,
            deferred_ci_percentile=known.deferred_ci_percentile,
            deferred_weight=known.deferred_weight,
            soc_guard_threshold=known.soc_guard_threshold,
            low_term_threshold=known.low_term_threshold,
            guard_penalty_mult=known.guard_penalty_mult,
        ),
        algo_name="Sundial-MPC",
        description=(
            "Sundial-MPC — weighted-sum with deferred battery carbon, "
            "CI-adaptive mode selection, native 15-min, CI-only TSFM"
        ),
        extra_args_fn=_add_mpc_args,
    )
