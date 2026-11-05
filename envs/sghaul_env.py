"""Ambiente Gymnasium para slicing URLLC/eMBB/mMTC no uplink SGHaul."""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.buffer_manager import UplinkBuffer5G
from envs.wisun_emulator import WiSUN_Emulator


class SGHaulEnv(gym.Env):
    metadata = {"render_modes": []}

    # Ordem fixa das 25 variáveis observadas (documentada para a heurística).
    OBS_INDEX = {
        "backlog_after": slice(0, 3),
        "backlog_before_service": slice(3, 6),
        "arrivals_bytes": slice(6, 9),
        "arrivals_packets": slice(9, 12),
        "min_slack_before": slice(12, 15),
        "mean_slack_before": slice(15, 18),
        "hol_after": slice(18, 21),
        "sinr_normalized": 21,
        "previous_action": slice(22, 25),
    }

    def __init__(
        self,
        w1: float = 10.0,
        w2: float = 10.0,
        w3: float = 5.0,
        w4: float = 5.0,
        drop_penalty_weight: float = 5.0,
        rho_target: Optional[float] = 0.95,
        traffic_scale: float = 1.0,
        bandwidth_hz: float = 1e6,
        step_duration: float = 0.01,
        mean_sinr_db: float = 10.0,
        allow_embb_borrowing: bool = True,
        mmpp_enabled: bool = True,
        calibration_path: Optional[str] = None,
        payload_overhead_bytes: int = 60,
        max_steps: int = 50_000,
    ):
        super().__init__()
        if step_duration <= 0 or bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz e step_duration devem ser positivos.")

        self.bandwidth_hz = float(bandwidth_hz)
        self.step_duration = float(step_duration)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps deve ser positivo.")
        self.mean_sinr_db = float(mean_sinr_db)
        self.rho_target = rho_target
        self.base_traffic_scale = float(traffic_scale)
        self.allow_embb_borrowing = bool(allow_embb_borrowing)
        self.mmpp_enabled = bool(mmpp_enabled)
        self.calibration_path = calibration_path
        self.payload_overhead_bytes = int(max(0, payload_overhead_bytes))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        low = np.zeros(25, dtype=np.float32)
        high = np.full(25, 10.0, dtype=np.float32)
        low[12:18] = -1.0  # slack pode ser negativo antes do descarte por deadline
        high[12:18] = 1.0
        low[22:25] = -1.0
        high[22:25] = 1.0
        low[21] = 0.0
        high[21] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.w1 = float(w1)  # throughput eMBB
        self.w2 = float(w2)  # violações URLLC
        self.w3 = float(w3)  # violações eMBB
        self.w4 = float(w4)  # violações mMTC
        self.drop_penalty_weight = float(drop_penalty_weight)

        self.topologia_atual = "Linear"
        self.current_time = 0.0
        self.current_step = 0
        self.sinr_db_atual = self.mean_sinr_db
        self.acao_anterior = np.zeros(3, dtype=np.float32)
        self.state = np.zeros(25, dtype=np.float32)
        self.wisun = None
        self.buffer = UplinkBuffer5G(allow_embb_borrowing=self.allow_embb_borrowing)
        self.last_rho_target = self.rho_target
        self.last_traffic_scale = self.base_traffic_scale

    def _create_wisun(self) -> WiSUN_Emulator:
        return WiSUN_Emulator(
            topology_name=self.topologia_atual,
            rng=self.np_random,
            calibration_path=self.calibration_path,
            payload_overhead_bytes=self.payload_overhead_bytes,
            traffic_scale=self.base_traffic_scale,
            rho_target=self.rho_target,
            bandwidth_hz=self.bandwidth_hz,
            mean_sinr_db=self.mean_sinr_db,
            mmpp_enabled=self.mmpp_enabled,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        topologias = ("Linear", "Mesh", "Quadrada")
        requested_topology = options.get("topologia")
        if requested_topology is not None:
            if requested_topology not in topologias:
                raise ValueError(f"Topologia inválida: {requested_topology}")
            self.topologia_atual = requested_topology
        else:
            self.topologia_atual = str(self.np_random.choice(topologias))

        if "rho_target" in options:
            self.rho_target = float(options["rho_target"])

        self.wisun = self._create_wisun()
        self.buffer = UplinkBuffer5G(allow_embb_borrowing=self.allow_embb_borrowing)
        self.current_time = 0.0
        self.current_step = 0
        self.sinr_db_atual = self.mean_sinr_db
        self.acao_anterior = np.zeros(3, dtype=np.float32)
        self.last_rho_target = self.rho_target
        self.last_traffic_scale = self.wisun.traffic_scale
        self.state = np.zeros(25, dtype=np.float32)
        return self.state.copy(), {"topologia": self.topologia_atual, "rho_target": self.rho_target}

    def _gerar_sinr_ar1(self) -> float:
        """AR(1) estacionário em dB, com média explicitamente controlada."""
        rho = 0.9
        sigma = 2.0
        noise = self.np_random.normal(0.0, sigma)
        self.sinr_db_atual = (
            self.mean_sinr_db + rho * (self.sinr_db_atual - self.mean_sinr_db) + noise
        )
        sinr_linear = 10.0 ** (self.sinr_db_atual / 10.0)
        return float(max(0.1, sinr_linear))

    @staticmethod
    def _safe_ratio(value: float, denominator: float, cap: float = 10.0) -> float:
        return float(np.clip(value / max(denominator, 1e-9), 0.0, cap))

    def _build_observation(self, before_metrics: dict, after_metrics: dict, arrivals: dict, sinr_db: float):
        names = ("UC16", "UC3", "MMTC")
        max_buffer = float(self.buffer.max_buffer_bytes)
        pdbs = {"UC16": self.buffer.pdb_uc16, "UC3": self.buffer.pdb_uc3, "MMTC": self.buffer.pdb_mmtc}

        values = []
        # Backlog depois do serviço.
        values.extend(self._safe_ratio(after_metrics[n]["bytes"], max_buffer, 1.0) for n in names)
        # Backlog imediatamente antes do serviço (inclui as chegadas daquele TTI).
        values.extend(self._safe_ratio(before_metrics[n]["bytes"], max_buffer, 10.0) for n in names)
        # Bytes e pacotes que chegaram ao gateway naquele TTI; inclui os elegíveis antes do DOA.
        values.extend(self._safe_ratio(arrivals["bytes"][n], max_buffer, 10.0) for n in names)
        values.extend(self._safe_ratio(arrivals["packets"][n], 100.0, 10.0) for n in names)
        # Slack antes do serviço, mínimo e médio, já descontando a espera na fila.
        values.extend(float(np.clip(before_metrics[n]["min_slack_ms"] / pdbs[n], -1.0, 1.0)) for n in names)
        values.extend(float(np.clip(before_metrics[n]["mean_slack_ms"] / pdbs[n], -1.0, 1.0)) for n in names)
        # HOL depois do serviço, para a próxima decisão.
        values.extend(self._safe_ratio(after_metrics[n]["hol_delay_ms"], pdbs[n], 10.0) for n in names)
        # SINR em dB normalizado no intervalo [-10, 30] dB.
        values.append(float(np.clip((sinr_db + 10.0) / 40.0, 0.0, 1.0)))
        values.extend(float(np.clip(x, -1.0, 1.0)) for x in self.acao_anterior)
        return np.asarray(values, dtype=np.float32)

    def step(self, action):
        if self.wisun is None:
            raise RuntimeError("É necessário chamar reset() antes de step().")

        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.current_step += 1

        # Softmax estável. O piso garante pelo menos 2% para cada slice.
        logits = 2.0 * action.astype(np.float64)
        logits -= np.max(logits)
        exp_action = np.exp(logits)
        softmax = exp_action / np.sum(exp_action)
        allocation = 0.02 + 0.94 * softmax
        rho_url, rho_embb, rho_mmtc = allocation

        sinr_linear = self._gerar_sinr_ar1()
        total_capacity_bytes = (
            self.bandwidth_hz * np.log2(1.0 + sinr_linear) * self.step_duration / 8.0
        )
        cap_url = total_capacity_bytes * rho_url
        cap_embb = total_capacity_bytes * rho_embb
        cap_mmtc = total_capacity_bytes * rho_mmtc

        packets = self.wisun.step(self.current_time, self.step_duration)
        doa_drops, buffer_drops, arrived = self.buffer.receive_packets(packets, self.current_time)

        # Estado de demanda capturado ANTES do serviço para compor a próxima observação.
        before_metrics = self.buffer.get_queue_metrics(self.current_time)
        arrivals = self.buffer.last_arrivals

        # Serviço em fronteira de TTI. As chegadas do TTI têm timestamp=current_time,
        # portanto não recebem uma penalidade artificial U(0, 10 ms).
        stats = self.buffer.serve_queues(cap_url, cap_embb, cap_mmtc, self.current_time)
        after_metrics = self.buffer.get_queue_metrics(self.current_time)

        sent_url = stats["throughput_uc16_bytes"]
        sent_embb = stats["throughput_uc3_bytes"]
        sent_mmtc = stats["throughput_mmtc_bytes"]
        thp_url = sent_url * 8.0 / (self.step_duration * 1e6)
        thp_embb = sent_embb * 8.0 / (self.step_duration * 1e6)
        thp_mmtc = sent_mmtc * 8.0 / (self.step_duration * 1e6)

        # Violações por classe incluem pacotes expirados e descartes por overflow.
        # Métricas de SLA incluem expiração e overflow. A recompensa separa os
        # eventos: deadline é penalizado por classe e overflow uma única vez abaixo.
        overflow_url = buffer_drops["UC16"]
        overflow_embb = buffer_drops["UC3"]
        overflow_mmtc = buffer_drops["UC1"] + buffer_drops["UC17"]
        viol_url = stats["pdb_violations_uc16"] + overflow_url
        viol_embb = stats["pdb_violations_uc3"] + overflow_embb
        viol_mmtc = stats["pdb_violations_mmtc"] + overflow_mmtc

        den_url = max(1, before_metrics["UC16"]["packets"] + stats["pdb_violations_uc16"] + overflow_url)
        den_embb = max(1, before_metrics["UC3"]["packets"] + stats["pdb_violations_uc3"] + overflow_embb)
        den_mmtc = max(1, before_metrics["MMTC"]["packets"] + stats["pdb_violations_mmtc"] + overflow_mmtc)
        rate_url = min(1.0, stats["pdb_violations_uc16"] / den_url)
        rate_embb = min(1.0, stats["pdb_violations_uc3"] / den_embb)
        rate_mmtc = min(1.0, stats["pdb_violations_mmtc"] / den_mmtc)
        total_arrivals = max(1, sum(arrived.values()))
        drop_rate = min(1.0, sum(buffer_drops.values()) / total_arrivals)

        normalized_embb_throughput = float(
            np.clip(sent_embb / max(total_capacity_bytes, 1e-9), 0.0, 1.0)
        )
        active_slacks = []
        for name in ("UC16", "UC3", "MMTC"):
            if before_metrics[name]["packets"] > 0:
                active_slacks.append(
                    np.clip(before_metrics[name]["mean_slack_ms"] / self.buffer._pdb(name), -1.0, 1.0)
                )
        slack_bonus = 0.1 * float(np.mean(active_slacks)) if active_slacks else 0.0

        # w1=throughput eMBB; w2=violações URLLC; w3=violações eMBB;
        # w4=violações mMTC. Drops têm peso independente.
        reward = (
            self.w1 * normalized_embb_throughput
            - self.w2 * rate_url
            - self.w3 * rate_embb
            - self.w4 * rate_mmtc
            - self.drop_penalty_weight * drop_rate
            + slack_bonus
        )
        reward = float(np.clip(reward, -20.0, 20.0))

        generated = self.wisun.last_step_stats["gerados"]
        pdr_drops = self.wisun.last_step_stats["falhas_pdr"]
        total_pdr_drops = sum(pdr_drops.values())
        total_buffer_drops = sum(buffer_drops.values())
        pending = self.buffer.count_pending_packets()

        self.acao_anterior = action.copy()
        self.state = self._build_observation(before_metrics, after_metrics, arrivals, self.sinr_db_atual)
        self.current_time += self.step_duration

        terminated = False
        truncated = bool(self.current_step >= self.max_steps)
        allocated_url_mbps = cap_url * 8.0 / (self.step_duration * 1e6)
        allocated_embb_mbps = cap_embb * 8.0 / (self.step_duration * 1e6)
        allocated_mmtc_mbps = cap_mmtc * 8.0 / (self.step_duration * 1e6)

        # Métricas de latência são expostas como soma e contagem para agregação ponderada correta.
        info = {
            "topologia": self.topologia_atual,
            "rho_target": self.rho_target,
            "rho_realized": float(arrivals["bytes"]["UC16"] + arrivals["bytes"]["UC3"] + arrivals["bytes"]["MMTC"]) / max(total_capacity_bytes, 1e-9),
            "traffic_scale": self.wisun.traffic_scale,
            "sinr_db": self.sinr_db_atual,
            "throughput_total_mbps": thp_url + thp_embb + thp_mmtc,
            "throughput_urllc_mbps": thp_url,
            "throughput_embb_mbps": thp_embb,
            "throughput_mmtc_mbps": thp_mmtc,
            "alocado_urllc_mbps": allocated_url_mbps,
            "alocado_embb_mbps": allocated_embb_mbps,
            "alocado_mmtc_mbps": allocated_mmtc_mbps,
            "fila_urllc_kb": after_metrics["UC16"]["bytes"] / 1024.0,
            "fila_embb_kb": after_metrics["UC3"]["bytes"] / 1024.0,
            "fila_mmtc_kb": after_metrics["MMTC"]["bytes"] / 1024.0,
            "drops_buffer": total_buffer_drops,
            "doa_uc16": doa_drops["UC16"],
            "doa_uc3": doa_drops["UC3"],
            "doa_mmtc": doa_drops["UC1"] + doa_drops["UC17"],
            "pdr_drops_uc16": pdr_drops["UC16"],
            "pdr_drops_uc3": pdr_drops["UC3"],
            "pdr_drops_mmtc": pdr_drops["UC1"] + pdr_drops["UC17"],
            "generated_uc16": generated["UC16"],
            "generated_uc3": generated["UC3"],
            "generated_mmtc": generated["UC1"] + generated["UC17"],
            "arrivals_uc16": arrived["UC16"],
            "arrivals_uc3": arrived["UC3"],
            "arrivals_mmtc": arrived["UC1"] + arrived["UC17"],
            "violacoes_uc16": viol_url,
            "violacoes_uc3": viol_embb,
            "violacoes_mmtc": viol_mmtc,
            "total_pacotes_uc16": arrived["UC16"] - doa_drops["UC16"],
            "total_pacotes_uc3": arrived["UC3"] - doa_drops["UC3"],
            "total_pacotes_mmtc": arrived["UC1"] + arrived["UC17"] - doa_drops["UC1"] - doa_drops["UC17"],
            "avg_latency_uc16": stats["avg_latency_uc16"],
            "avg_latency_uc3": stats["avg_latency_uc3"],
            "avg_latency_mmtc": stats["avg_latency_mmtc"],
            "latency_sum_uc16": stats["latency_sum_uc16"],
            "latency_sum_uc3": stats["latency_sum_uc3"],
            "latency_sum_mmtc": stats["latency_sum_mmtc"],
            "latency_count_uc16": stats["latency_count_uc16"],
            "latency_count_uc3": stats["latency_count_uc3"],
            "latency_count_mmtc": stats["latency_count_mmtc"],
            "avg_wisun_delay_uc16": stats["avg_wisun_delay_uc16"],
            "avg_wisun_delay_uc3": stats["avg_wisun_delay_uc3"],
            "avg_wisun_delay_mmtc": stats["avg_wisun_delay_mmtc"],
            "hol_delay_uc16": after_metrics["UC16"]["hol_delay_ms"],
            "hol_delay_uc3": after_metrics["UC3"]["hol_delay_ms"],
            "hol_delay_mmtc": after_metrics["MMTC"]["hol_delay_ms"],
            "pending_uc16": pending["UC16"],
            "pending_uc3": pending["UC3"],
            "pending_mmtc": pending["MMTC"],
            "mmpp_state": self.wisun.mmpp_state,
        }

        return self.state.copy(), reward, terminated, truncated, info
