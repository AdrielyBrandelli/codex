"""Emulador de tráfego Wi-SUN para o ambiente SGHaul.

- Usa um numpy.random.Generator próprio (injetado pelo ambiente Gymnasium).
- Carrega calibração por caminho explícito e avisa quando usa defaults.
- Modela tempestades por MMPP de dois estados (normal/tempestade).
- Permite calibrar a carga oferecida para uma razão carga/capacidade (rho_target).
- Mantém os tamanhos de payload separados do overhead configurável.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Packet:
    id_pacote: int
    tipo_trafego: str
    tamanho_bytes: float
    hop_count: int
    d_wisun: float
    timestamp_chegada: float


class WiSUN_Emulator:
    TRAFFIC_TYPES = ("UC16", "UC3", "UC1", "UC17")
    _warned_missing_calibration = False

    def __init__(
        self,
        topology_name: str = "Linear",
        lambda_uc16: float = 10.0,
        lambda_uc1: float = 20.0,
        lambda_uc17: float = 30.0,
        burst_uc3: float = 50.0,
        periodo_uc3: float = 5.0,
        max_hops: int = 10,
        rng: Optional[np.random.Generator] = None,
        calibration_path: Optional[str] = None,
        payload_overhead_bytes: int = 60,
        traffic_scale: float = 1.0,
        rho_target: Optional[float] = None,
        bandwidth_hz: float = 1e6,
        mean_sinr_db: float = 10.0,
        mmpp_enabled: bool = True,
        p_normal_to_storm: float = 0.005,
        p_storm_to_normal: float = 0.05,
        storm_multiplier_uc16: float = 20.0,
        storm_multiplier_uc3: float = 3.0,
    ):
        self.topology_name = topology_name
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_hops = int(max(1, max_hops))
        self.payload_overhead_bytes = int(max(0, payload_overhead_bytes))
        self.bandwidth_hz = float(bandwidth_hz)
        self.mean_sinr_db = float(mean_sinr_db)

        self.lambda_uc16 = float(lambda_uc16)
        self.lambda_uc1 = float(lambda_uc1)
        self.lambda_uc17 = float(lambda_uc17)
        self.burst_uc3 = float(burst_uc3)
        self.periodo_uc3 = float(periodo_uc3)

        self.alpha_uc16 = 20.0
        self.alpha_uc3 = 35.0
        self.alpha_uc1 = 10.0
        self.alpha_uc17 = 10.0
        self.beta = 10.0
        self.std_dev_jitter = 5.0
        self.pdr_slope = -0.015
        self.pdr_intercept = 1.0

        # Payloads de aplicação; o overhead é adicionado uma única vez abaixo.
        self.payload_uc16 = 100.0
        self.payload_uc3 = 400.0
        self.payload_uc1 = 150.0
        self.payload_uc17 = 150.0
        self.tamanho_uc16 = self.payload_uc16 + self.payload_overhead_bytes
        self.tamanho_uc3 = self.payload_uc3 + self.payload_overhead_bytes
        self.tamanho_uc1 = self.payload_uc1 + self.payload_overhead_bytes
        self.tamanho_uc17 = self.payload_uc17 + self.payload_overhead_bytes

        self.mmpp_enabled = bool(mmpp_enabled)
        self.p_normal_to_storm = float(np.clip(p_normal_to_storm, 0.0, 1.0))
        self.p_storm_to_normal = float(np.clip(p_storm_to_normal, 0.0, 1.0))
        self.storm_multiplier_uc16 = float(max(1.0, storm_multiplier_uc16))
        self.storm_multiplier_uc3 = float(max(1.0, storm_multiplier_uc3))
        self.mmpp_state = 0  # 0 = normal; 1 = tempestade

        self.traffic_scale = float(max(0.0, traffic_scale))
        self.rho_target = rho_target
        self.packet_counter = 0
        self.last_uc3_burst_time = -self.periodo_uc3
        self.last_step_stats = self._empty_step_stats()

        self._load_calibration(calibration_path)
        self._refresh_packet_sizes()
        if rho_target is not None:
            if not (0.0 < float(rho_target) < 5.0):
                raise ValueError("rho_target deve estar entre 0 e 5.")
            estimated_load = self.estimate_offered_load_mbps()
            reference_capacity = self._reference_capacity_mbps()
            if estimated_load <= 0.0:
                raise ValueError("Não é possível calibrar rho: carga oferecida estimada é zero.")
            self.traffic_scale *= float(rho_target) * reference_capacity / estimated_load

    @staticmethod
    def _empty_step_stats() -> Dict[str, Dict[str, int]]:
        keys = ("gerados", "falhas_pdr")
        return {k: {uc: 0 for uc in ("UC16", "UC3", "UC1", "UC17")} for k in keys}

    def _default_calibration_path(self) -> Path:
        # Este arquivo fica em <projeto>/envs/wisun_emulator.py.
        return Path(__file__).resolve().parents[1] / "data" / "calibracao.json"

    def _load_calibration(self, calibration_path: Optional[str]) -> None:
        path = Path(calibration_path).expanduser() if calibration_path else self._default_calibration_path()
        if not path.exists():
            if not WiSUN_Emulator._warned_missing_calibration:
                warnings.warn(
                    f"Calibração não encontrada em '{path}'. Serão usados parâmetros padrão.",
                    RuntimeWarning,
                )
                WiSUN_Emulator._warned_missing_calibration = True
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                calibracoes = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Não foi possível ler a calibração '{path}': {exc}") from exc

        if self.topology_name not in calibracoes:
            warnings.warn(
                f"Topologia '{self.topology_name}' não existe em '{path}'. Serão usados defaults.",
                RuntimeWarning,
            )
            return

        dados = calibracoes[self.topology_name]
        fields = (
            "alpha_uc16", "alpha_uc3", "alpha_uc1", "alpha_uc17", "beta",
            "std_dev_jitter", "pdr_slope", "pdr_intercept", "lambda_uc16",
            "lambda_uc1", "lambda_uc17", "burst_uc3", "periodo_uc3",
        )
        for field in fields:
            if field in dados and dados[field] is not None:
                value = float(dados[field])
                if not math.isfinite(value):
                    raise ValueError(f"Valor não finito em calibracao.json: {self.topology_name}.{field}")
                setattr(self, field, value)

        # O JSON grava tamanho_uc* como payload de aplicação (antes do overhead).
        for uc in ("uc16", "uc3", "uc1", "uc17"):
            field = f"tamanho_{uc}"
            if field in dados and dados[field] is not None:
                value = float(dados[field])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{field} inválido em calibracao.json para {self.topology_name}.")
                setattr(self, f"payload_{uc}", value)

        if self.periodo_uc3 <= 0:
            raise ValueError("periodo_uc3 deve ser maior que zero.")
        for field in ("lambda_uc16", "lambda_uc1", "lambda_uc17", "burst_uc3"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} não pode ser negativo.")
        for field in ("alpha_uc16", "alpha_uc3", "alpha_uc1", "alpha_uc17", "std_dev_jitter"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} não pode ser negativo.")
        self.pdr_slope = float(self.pdr_slope)
        self.pdr_intercept = float(self.pdr_intercept)

    def _refresh_packet_sizes(self) -> None:
        self.tamanho_uc16 = self.payload_uc16 + self.payload_overhead_bytes
        self.tamanho_uc3 = self.payload_uc3 + self.payload_overhead_bytes
        self.tamanho_uc1 = self.payload_uc1 + self.payload_overhead_bytes
        self.tamanho_uc17 = self.payload_uc17 + self.payload_overhead_bytes

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def _mean_pdr(self) -> float:
        probs = [
            np.clip(self.pdr_slope * hop + self.pdr_intercept, 0.0, 1.0)
            for hop in range(1, self.max_hops + 1)
        ]
        return float(np.mean(probs))

    def _storm_stationary_probability(self) -> float:
        denom = self.p_normal_to_storm + self.p_storm_to_normal
        return self.p_normal_to_storm / denom if denom > 0 else 0.0

    def estimate_offered_load_mbps(self) -> float:
        """Carga média estimada após PDR, incluindo a média estacionária do MMPP."""
        pdr = self._mean_pdr()
        p_storm = self._storm_stationary_probability() if self.mmpp_enabled else 0.0
        mean_uc16_multiplier = 1.0 + p_storm * (self.storm_multiplier_uc16 - 1.0)
        mean_uc3_multiplier = 1.0 + p_storm * (self.storm_multiplier_uc3 - 1.0)

        packets_per_second = {
            "UC16": self.lambda_uc16 * mean_uc16_multiplier * pdr,
            "UC1": self.lambda_uc1 * pdr,
            "UC17": self.lambda_uc17 * pdr,
            "UC3": (self.burst_uc3 / self.periodo_uc3) * mean_uc3_multiplier * pdr,
        }
        bits_per_second = (
            packets_per_second["UC16"] * self.tamanho_uc16
            + packets_per_second["UC1"] * self.tamanho_uc1
            + packets_per_second["UC17"] * self.tamanho_uc17
            + packets_per_second["UC3"] * self.tamanho_uc3
        ) * 8.0
        return float(bits_per_second / 1e6)

    def _reference_capacity_mbps(self) -> float:
        sinr_linear = 10.0 ** (self.mean_sinr_db / 10.0)
        return float(self.bandwidth_hz * np.log2(1.0 + sinr_linear) / 1e6)

    def _calcular_atraso(self, hop_count: int, tipo_trafego: str) -> float:
        alpha = {
            "UC16": self.alpha_uc16,
            "UC3": self.alpha_uc3,
            "UC1": self.alpha_uc1,
            "UC17": self.alpha_uc17,
        }.get(tipo_trafego, self.alpha_uc1)
        epsilon = self.rng.normal(loc=0.0, scale=self.std_dev_jitter)
        return float(max(1.0, alpha * hop_count + self.beta + epsilon))

    def _simular_pdr(self, hop_count: int) -> bool:
        probability = np.clip(self.pdr_slope * hop_count + self.pdr_intercept, 0.0, 1.0)
        return bool(self.rng.random() <= probability)

    def _advance_mmpp(self) -> int:
        if not self.mmpp_enabled:
            self.mmpp_state = 0
            return 0
        if self.mmpp_state == 0:
            if self.rng.random() < self.p_normal_to_storm:
                self.mmpp_state = 1
        elif self.rng.random() < self.p_storm_to_normal:
            self.mmpp_state = 0
        return self.mmpp_state

    def _stochastic_round(self, value: float) -> int:
        base = int(math.floor(value))
        return base + int(self.rng.random() < (value - base))

    def step(self, current_time: float, step_duration: float) -> List[Packet]:
        if step_duration <= 0:
            raise ValueError("step_duration deve ser positivo.")

        packets: List[Packet] = []
        self.last_step_stats = self._empty_step_stats()
        storm_state = self._advance_mmpp()
        mult_uc16 = self.storm_multiplier_uc16 if storm_state else 1.0
        mult_uc3 = self.storm_multiplier_uc3 if storm_state else 1.0

        # Modelo discreto por TTI: as chegadas são posicionadas na fronteira do step.
        # Isso evita adicionar artificialmente U(0, TTI) ao atraso de fila.
        arrival_time = float(current_time)

        # UC16: alarmes/eventos com taxa modulada pelo estado MMPP.
        count_uc16 = int(self.rng.poisson(self.lambda_uc16 * step_duration * self.traffic_scale * mult_uc16))
        self.last_step_stats["gerados"]["UC16"] = count_uc16
        for _ in range(count_uc16):
            hops = int(self.rng.integers(1, self.max_hops + 1))
            if not self._simular_pdr(hops):
                self.last_step_stats["falhas_pdr"]["UC16"] += 1
                continue
            self.packet_counter += 1
            packets.append(Packet(self.packet_counter, "UC16", self.tamanho_uc16, hops,
                                  self._calcular_atraso(hops, "UC16"), arrival_time))

        # UC3: rajada periódica; tempestades ampliam rajadas coincidentes.
        if (current_time - self.last_uc3_burst_time) >= self.periodo_uc3 - 1e-12:
            self.last_uc3_burst_time = float(current_time)
            batch_size = self._stochastic_round(self.burst_uc3 * self.traffic_scale * mult_uc3)
            self.last_step_stats["gerados"]["UC3"] = batch_size
            for _ in range(batch_size):
                hops = int(self.rng.integers(1, self.max_hops + 1))
                if not self._simular_pdr(hops):
                    self.last_step_stats["falhas_pdr"]["UC3"] += 1
                    continue
                self.packet_counter += 1
                packets.append(Packet(self.packet_counter, "UC3", self.tamanho_uc3, hops,
                                      self._calcular_atraso(hops, "UC3"), arrival_time))

        # UC1 e UC17: tráfego de manutenção/registro e disponibilidade.
        for uc, rate, size, alpha_type in (
            ("UC1", self.lambda_uc1, self.tamanho_uc1, "UC1"),
            ("UC17", self.lambda_uc17, self.tamanho_uc17, "UC17"),
        ):
            count = int(self.rng.poisson(rate * step_duration * self.traffic_scale))
            self.last_step_stats["gerados"][uc] = count
            for _ in range(count):
                hops = int(self.rng.integers(1, self.max_hops + 1))
                if not self._simular_pdr(hops):
                    self.last_step_stats["falhas_pdr"][uc] += 1
                    continue
                self.packet_counter += 1
                packets.append(Packet(self.packet_counter, uc, size, hops,
                                      self._calcular_atraso(hops, alpha_type), arrival_time))

        return packets
