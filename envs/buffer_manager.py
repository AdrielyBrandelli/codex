"""Buffers por slice, descarte por deadline e serviço com segmentação em bytes."""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Tuple

from envs.wisun_emulator import Packet


class UplinkBuffer5G:
    def __init__(
        self,
        pdb_uc16: float = 300.0,
        pdb_uc3: float = 2000.0,
        pdb_mmtc: float = 5000.0,
        max_buffer_bytes: int = 512 * 1024,
        allow_embb_borrowing: bool = True,
    ):
        self.queue_uc16: Deque[dict] = deque()
        self.queue_uc3: Deque[dict] = deque()
        self.queue_mmtc: Deque[dict] = deque()
        self.pdb_uc16 = float(pdb_uc16)
        self.pdb_uc3 = float(pdb_uc3)
        self.pdb_mmtc = float(pdb_mmtc)
        self.max_buffer_bytes = int(max_buffer_bytes)
        self.allow_embb_borrowing = bool(allow_embb_borrowing)
        self.bytes_uc16 = 0.0
        self.bytes_uc3 = 0.0
        self.bytes_mmtc = 0.0
        self.last_arrivals = self._empty_arrival_stats()
        self._arrival_expired = {"UC16": 0, "UC3": 0, "MMTC": 0}

    @staticmethod
    def _empty_arrival_stats() -> dict:
        return {
            "bytes": {"UC16": 0.0, "UC3": 0.0, "MMTC": 0.0},
            "packets": {"UC16": 0, "UC3": 0, "MMTC": 0},
            "accepted_bytes": {"UC16": 0.0, "UC3": 0.0, "MMTC": 0.0},
            "accepted_packets": {"UC16": 0, "UC3": 0, "MMTC": 0},
        }

    def _classify(self, traffic_type: str) -> str:
        if traffic_type == "UC16":
            return "UC16"
        if traffic_type == "UC3":
            return "UC3"
        if traffic_type in ("UC1", "UC17"):
            return "MMTC"
        raise ValueError(f"Tipo de tráfego desconhecido: {traffic_type}")

    def _queue_and_bytes(self, slice_name: str):
        if slice_name == "UC16":
            return self.queue_uc16, self.bytes_uc16
        if slice_name == "UC3":
            return self.queue_uc3, self.bytes_uc3
        return self.queue_mmtc, self.bytes_mmtc

    def receive_packets(self, packets: List[Packet], current_time: float):
        doa_drops = {"UC16": 0, "UC3": 0, "UC1": 0, "UC17": 0}
        buffer_drops = {"UC16": 0, "UC3": 0, "UC1": 0, "UC17": 0}
        arrived = {"UC16": 0, "UC3": 0, "UC1": 0, "UC17": 0}
        self.last_arrivals = self._empty_arrival_stats()
        self._arrival_expired = {"UC16": 0, "UC3": 0, "MMTC": 0}

        # Expira antes de admitir chegadas novas, para não rejeitar pacote válido
        # por bytes ocupados por pacotes cujo prazo já acabou.
        for name, queue in (("UC16", self.queue_uc16), ("UC3", self.queue_uc3), ("MMTC", self.queue_mmtc)):
            expired, _ = self._expire_queue(queue, name, current_time)
            self._arrival_expired[name] += expired

        for packet in packets:
            uc = packet.tipo_trafego
            slice_name = self._classify(uc)
            arrived[uc] += 1
            self.last_arrivals["bytes"][slice_name] += float(packet.tamanho_bytes)
            self.last_arrivals["packets"][slice_name] += 1

            pdb = self.pdb_uc16 if slice_name == "UC16" else self.pdb_uc3 if slice_name == "UC3" else self.pdb_mmtc
            if packet.d_wisun > pdb:
                doa_drops[uc] += 1
                continue

            queue, used_bytes = self._queue_and_bytes(slice_name)
            used_bytes = self._queue_bytes_counter(slice_name)
            if used_bytes + packet.tamanho_bytes > self.max_buffer_bytes:
                buffer_drops[uc] += 1
                continue

            queue.append({
                "packet": packet,
                "arrival_time": float(packet.timestamp_chegada),
                "remaining_bytes": float(packet.tamanho_bytes),
            })
            self._set_bytes(slice_name, used_bytes + float(packet.tamanho_bytes))
            self.last_arrivals["accepted_bytes"][slice_name] += float(packet.tamanho_bytes)
            self.last_arrivals["accepted_packets"][slice_name] += 1

        return doa_drops, buffer_drops, arrived

    def _queue_bytes_counter(self, slice_name: str) -> float:
        if slice_name == "UC16":
            return self.bytes_uc16
        if slice_name == "UC3":
            return self.bytes_uc3
        return self.bytes_mmtc

    def _set_bytes(self, slice_name: str, value: float) -> None:
        if slice_name == "UC16":
            self.bytes_uc16 = float(value)
        elif slice_name == "UC3":
            self.bytes_uc3 = float(value)
        else:
            self.bytes_mmtc = float(value)

    def _pdb(self, slice_name: str) -> float:
        return self.pdb_uc16 if slice_name == "UC16" else self.pdb_uc3 if slice_name == "UC3" else self.pdb_mmtc

    def _expire_queue(self, queue: Deque[dict], slice_name: str, current_time: float) -> Tuple[int, int]:
        """Remove todos os pacotes que já ultrapassaram o PDB E2E."""
        kept = deque()
        expired_packets = 0
        expired_bytes = 0.0
        pdb = self._pdb(slice_name)
        for item in queue:
            packet = item["packet"]
            queue_wait_ms = max(0.0, current_time - item["arrival_time"]) * 1000.0
            if packet.d_wisun + queue_wait_ms > pdb:
                expired_packets += 1
                expired_bytes += item["remaining_bytes"]
            else:
                kept.append(item)
        queue.clear()
        queue.extend(kept)
        self._set_bytes(slice_name, max(0.0, self._queue_bytes_counter(slice_name) - expired_bytes))
        return expired_packets, int(round(expired_bytes))

    @staticmethod
    def _queue_bytes(queue: Iterable[dict]) -> float:
        return float(sum(item["remaining_bytes"] for item in queue))

    def _serve_one_queue(self, queue: Deque[dict], slice_name: str, capacity_bytes: float, current_time: float):
        sent_bytes = 0.0
        completed_packets = 0
        violations = 0
        latency_sum = 0.0
        latency_count = 0
        pdb = self._pdb(slice_name)
        remaining_capacity = max(0.0, float(capacity_bytes))

        while queue and remaining_capacity > 1e-9:
            item = queue[0]
            packet = item["packet"]
            amount = min(item["remaining_bytes"], remaining_capacity)
            item["remaining_bytes"] -= amount
            remaining_capacity -= amount
            sent_bytes += amount
            self._set_bytes(slice_name, max(0.0, self._queue_bytes_counter(slice_name) - amount))

            if item["remaining_bytes"] <= 1e-9:
                queue.popleft()
                completed_packets += 1
                latency = packet.d_wisun + max(0.0, current_time - item["arrival_time"]) * 1000.0
                latency_sum += latency
                latency_count += 1
                if latency > pdb:
                    violations += 1
            else:
                # FIFO: um pacote parcialmente transmitido mantém a cabeça da fila.
                break

        return {
            "bytes": sent_bytes,
            "completed": completed_packets,
            "violations": violations,
            "latency_sum": latency_sum,
            "latency_count": latency_count,
            "unused": remaining_capacity,
        }

    def serve_queues(
        self,
        capacity_uc16_bytes: float,
        capacity_uc3_bytes: float,
        capacity_mmtc_bytes: float,
        current_time: float,
    ) -> Dict[str, float]:
        stats = {
            "throughput_uc16_bytes": 0.0,
            "throughput_uc3_bytes": 0.0,
            "throughput_mmtc_bytes": 0.0,
            "pdb_violations_uc16": 0,
            "pdb_violations_uc3": 0,
            "pdb_violations_mmtc": 0,
            "expired_uc16": 0,
            "expired_uc3": 0,
            "expired_mmtc": 0,
            "completed_uc16": 0,
            "completed_uc3": 0,
            "completed_mmtc": 0,
            "latency_sum_uc16": 0.0,
            "latency_sum_uc3": 0.0,
            "latency_sum_mmtc": 0.0,
            "latency_count_uc16": 0,
            "latency_count_uc3": 0,
            "latency_count_mmtc": 0,
        }

        # Pacotes cujo prazo já venceu são removidos mesmo que ainda não possam ser servidos.
        for slice_name, queue in (("UC16", self.queue_uc16), ("UC3", self.queue_uc3), ("MMTC", self.queue_mmtc)):
            expired, _ = self._expire_queue(queue, slice_name, current_time)
            expired += self._arrival_expired.get(slice_name, 0)
            stats[f"expired_{slice_name.lower()}"] = expired
            stats[f"pdb_violations_{slice_name.lower()}"] = expired

        # URLLC e mMTC recebem suas fatias reservadas primeiro.
        r_url = self._serve_one_queue(self.queue_uc16, "UC16", capacity_uc16_bytes, current_time)
        r_mmtc = self._serve_one_queue(self.queue_mmtc, "MMTC", capacity_mmtc_bytes, current_time)

        # eMBB pode receber capacidade que ficou ociosa nas outras fatias.
        cap_embb = max(0.0, float(capacity_uc3_bytes))
        if self.allow_embb_borrowing and self.queue_uc3:
            cap_embb += r_url["unused"] + r_mmtc["unused"]
        r_embb = self._serve_one_queue(self.queue_uc3, "UC3", cap_embb, current_time)

        for suffix, result in (("uc16", r_url), ("uc3", r_embb), ("mmtc", r_mmtc)):
            stats[f"throughput_{suffix}_bytes"] = result["bytes"]
            stats[f"pdb_violations_{suffix}"] += result["violations"]
            stats[f"completed_{suffix}"] = result["completed"]
            stats[f"latency_sum_{suffix}"] = result["latency_sum"]
            stats[f"latency_count_{suffix}"] = result["latency_count"]
            stats[f"avg_latency_{suffix}"] = (
                result["latency_sum"] / result["latency_count"] if result["latency_count"] else 0.0
            )

        stats["avg_latency_uc16"] = stats["avg_latency_uc16"]
        stats["avg_latency_uc3"] = stats["avg_latency_uc3"]
        stats["avg_latency_mmtc"] = stats["avg_latency_mmtc"]
        stats["avg_wisun_delay_uc16"] = self._mean_wisun_delay(self.queue_uc16)
        stats["avg_wisun_delay_uc3"] = self._mean_wisun_delay(self.queue_uc3)
        stats["avg_wisun_delay_mmtc"] = self._mean_wisun_delay(self.queue_mmtc)
        stats["hol_delay_uc16"] = self._hol_delay(self.queue_uc16, current_time)
        stats["hol_delay_uc3"] = self._hol_delay(self.queue_uc3, current_time)
        stats["hol_delay_mmtc"] = self._hol_delay(self.queue_mmtc, current_time)
        stats["pending_packets_uc16"] = len(self.queue_uc16)
        stats["pending_packets_uc3"] = len(self.queue_uc3)
        stats["pending_packets_mmtc"] = len(self.queue_mmtc)
        return stats

    @staticmethod
    def _mean_wisun_delay(queue: Iterable[dict]) -> float:
        items = list(queue)
        return float(sum(i["packet"].d_wisun for i in items) / len(items)) if items else 0.0

    @staticmethod
    def _hol_delay(queue: Deque[dict], current_time: float) -> float:
        if not queue:
            return 0.0
        item = queue[0]
        return float(item["packet"].d_wisun + max(0.0, current_time - item["arrival_time"]) * 1000.0)

    def get_queue_sizes(self) -> tuple:
        return self.bytes_uc16, self.bytes_uc3, self.bytes_mmtc

    def get_queue_slacks(self, current_time: float = 0.0) -> tuple:
        metrics = self.get_queue_metrics(current_time)
        return (
            metrics["UC16"]["min_slack_ms"],
            metrics["UC3"]["min_slack_ms"],
            metrics["MMTC"]["min_slack_ms"],
        )

    def get_queue_metrics(self, current_time: float) -> Dict[str, dict]:
        result = {}
        for name, queue in (("UC16", self.queue_uc16), ("UC3", self.queue_uc3), ("MMTC", self.queue_mmtc)):
            pdb = self._pdb(name)
            slacks = []
            for item in queue:
                wait_ms = max(0.0, current_time - item["arrival_time"]) * 1000.0
                slacks.append(pdb - item["packet"].d_wisun - wait_ms)
            hol = self._hol_delay(queue, current_time)
            result[name] = {
                "bytes": self._queue_bytes_counter(name),
                "packets": len(queue),
                "min_slack_ms": float(min(slacks)) if slacks else pdb,
                "mean_slack_ms": float(sum(slacks) / len(slacks)) if slacks else pdb,
                "hol_delay_ms": hol,
            }
        return result

    def count_pending_packets(self) -> dict:
        return {
            "UC16": len(self.queue_uc16),
            "UC3": len(self.queue_uc3),
            "MMTC": len(self.queue_mmtc),
        }
