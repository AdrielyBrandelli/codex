"""Extrai pacotes TX/RX dos logs SGH/Lactec para data/dados_lactec_raw.csv.

O campo UDP SIZE, quando presente no log, tem prioridade. Na ausência dele,
usa-se o payload configurado por tipo de mensagem e registra-se a origem do tamanho.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd

CONFIG_TRAFEGO_SGH_LACTEC = {
    "event notification": {"uc": "UC16", "payload": 100},
    "event acknowledge": {"uc": "UC16", "payload": 100},
    "operation data request": {"uc": "UC3", "payload": 400},
    "operation data response": {"uc": "UC3", "payload": 400},
    "register request": {"uc": "UC1", "payload": 150},
    "association request": {"uc": "UC1", "payload": 150},
    "association response": {"uc": "UC1", "payload": 150},
    "disassociation request": {"uc": "UC1", "payload": 150},
    "disassociation response": {"uc": "UC1", "payload": 150},
    "availability request": {"uc": "UC17", "payload": 150},
    "availability response": {"uc": "UC17", "payload": 150},
}

FIELDNAMES = [
    "topologia", "tipo_trafego", "tamanho_bytes", "tamanho_origem",
    "hop_count", "latencia_ms", "entregue", "t0", "arquivo_origem",
]

RE_T0 = re.compile(r"\bt0\s*=\s*(\d+)", re.IGNORECASE)
RE_UDP_SIZE = re.compile(
    r"(?:UDP[_ ]?SIZE|UDP[_ ]?LENGTH|payload[_ ]?size|packet[_ ]?size)\s*[=:]\s*;?\s*(\d+)",
    re.IGNORECASE,
)
RE_LATENCY = re.compile(
    r"lat\s*\([^)]+\)\s*=\s*;?\s*([0-9]+(?:[.,][0-9]+)?)\s*;?\s*ms",
    re.IGNORECASE,
)
RE_ROUTE = re.compile(r"\$ROUTES\$(.*)")
RE_EXCEL_NODE = re.compile(r"\d+\.\d+\s+(\d+)\s+\[INFO", re.IGNORECASE)
RE_TXT_NODE = re.compile(r"^\d+\.\d+\s*;\s*(\d+)\s*;")


def _match_message(line: str) -> Optional[str]:
    low = line.lower()
    for key in sorted(CONFIG_TRAFEGO_SGH_LACTEC, key=len, reverse=True):
        if key in low:
            return key
    return None


def _extrair_dados_tx(line: str):
    """Retorna (UC, tamanho, t0, origem_tamanho) ou None."""
    message = _match_message(line)
    if message is None:
        return None

    match_t0 = RE_T0.search(line)
    if not match_t0:
        print(f"[ALERTA PARSER] Mensagem '{message}' sem t0: {line[:100]}...")
        return None

    match_size = RE_UDP_SIZE.search(line)
    if match_size:
        size = int(match_size.group(1))
        size_source = "UDP_SIZE_LOG"
        if size <= 0:
            print(f"[ALERTA PARSER] UDP SIZE inválido ({size}); usando payload configurado.")
            size = CONFIG_TRAFEGO_SGH_LACTEC[message]["payload"]
            size_source = "PAYLOAD_CONFIG"
    else:
        size = CONFIG_TRAFEGO_SGH_LACTEC[message]["payload"]
        size_source = "PAYLOAD_CONFIG"

    return CONFIG_TRAFEGO_SGH_LACTEC[message]["uc"], size, int(match_t0.group(1)), size_source


def _parse_routes(route_text: str) -> Dict[int, int]:
    routes: Dict[int, int] = {}
    for pair in route_text.split(";"):
        pair = pair.strip()
        if "-" not in pair:
            continue
        child_text, parent_text = pair.split("-", 1)
        try:
            routes[int(child_text.strip())] = int(parent_text.strip())
        except ValueError:
            continue
    return routes


def _count_hops(node_id: int, routes: Dict[int, int]) -> int:
    hops = 0
    current = int(node_id)
    visited = set()
    while current in routes and routes[current] != current and current not in visited:
        visited.add(current)
        current = routes[current]
        hops += 1
    return max(1, hops)


def _new_packet(topology: str, tx_data: tuple, node_id: int, routes: Dict[int, int], source: str) -> dict:
    uc, size, t0, size_source = tx_data
    return {
        "topologia": topology,
        "tipo_trafego": uc,
        "tamanho_bytes": int(size),
        "tamanho_origem": size_source,
        "hop_count": _count_hops(node_id, routes),
        "latencia_ms": 0.0,
        "entregue": 0,
        "t0": int(t0),
        "arquivo_origem": source,
    }


def _attach_rx(line: str, packets_by_t0: Dict[int, Deque[dict]]) -> None:
    match_t0 = RE_T0.search(line)
    match_latency = RE_LATENCY.search(line)
    if not match_t0 or not match_latency:
        return
    t0 = int(match_t0.group(1))
    queue = packets_by_t0.get(t0)
    if not queue:
        return
    packet = next((p for p in queue if p["entregue"] == 0), None)
    if packet is None:
        return
    packet["entregue"] = 1
    packet["latencia_ms"] = float(match_latency.group(1).replace(",", "."))


def parse_excel_linear(input_file: str) -> list:
    path = Path(input_file)
    print(f"Lendo Excel (Linear): {path}...")
    if not path.exists():
        print(f"[AVISO] Arquivo não encontrado: {path}. Linear será ignorada.")
        return []

    routes = _parse_routes(
        "0001-0001;0002-0001;0003-0002;0004-0003;0005-0004;0006-0005;"
        "0007-0006;0008-0007;0009-0008;0010-0009;0011-0010;"
    )
    try:
        frame = pd.read_excel(path, header=None)
    except Exception as exc:
        raise RuntimeError(f"Erro ao ler Excel '{path}': {exc}") from exc

    packets_by_t0: Dict[int, Deque[dict]] = defaultdict(deque)
    all_packets = []
    for _, row in frame.iterrows():
        line = " ".join(str(value) for value in row.values if pd.notna(value))
        if "UDP ROUTER TX" in line or "UDP BORDER TX" in line:
            tx_data = _extrair_dados_tx(line)
            node_match = RE_EXCEL_NODE.search(line)
            if tx_data and node_match:
                packet = _new_packet("Linear", tx_data, int(node_match.group(1)), routes, str(path))
                all_packets.append(packet)
                packets_by_t0[packet["t0"]].append(packet)
        elif "UDP BORDER RX" in line or "UDP ROUTER RX" in line:
            _attach_rx(line, packets_by_t0)
    return all_packets


def parse_txt_topology(diretorio_base: str, nome_topologia: str) -> list:
    base = Path(diretorio_base)
    print(f"Varrendo pasta ({nome_topologia}): {base}...")
    if not base.exists():
        print(f"[AVISO] Diretório não encontrado: {base}. {nome_topologia} será ignorada.")
        return []

    all_packets = []
    for filepath_text in sorted(glob.glob(str(base / "**" / "*.txt"), recursive=True)):
        filepath = Path(filepath_text)
        routes: Dict[int, int] = {}
        packets_by_t0: Dict[int, Deque[dict]] = defaultdict(deque)
        file_packets = []
        with filepath.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
        # Primeiro coleta todas as rotas; assim TX não depende da posição do registro $ROUTES$.
        for line in lines:
            if "$ROUTES$" in line:
                match = RE_ROUTE.search(line)
                if match:
                    routes.update(_parse_routes(match.group(1).strip()))
        # Segundo passe mantém a ordem TX/RX para associar entrega e latência.
        for line in lines:
            if "UDP ROUTER TX" in line or "UDP BORDER TX" in line:
                tx_data = _extrair_dados_tx(line)
                node_match = RE_TXT_NODE.search(line)
                if tx_data and node_match:
                    packet = _new_packet(
                        nome_topologia, tx_data, int(node_match.group(1)), routes, str(filepath)
                    )
                    file_packets.append(packet)
                    packets_by_t0[packet["t0"]].append(packet)
            elif "UDP BORDER RX" in line or "UDP ROUTER RX" in line:
                _attach_rx(line, packets_by_t0)
        all_packets.extend(file_packets)
    return all_packets


def consolidar_logs(
    excel_linear: str,
    dir_mesh: str,
    dir_quadrada: str,
    output_file: str,
) -> list:
    data = []
    data.extend(parse_excel_linear(excel_linear))
    data.extend(parse_txt_topology(dir_mesh, "Mesh"))
    data.extend(parse_txt_topology(dir_quadrada, "Quadrada"))

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(data)

    print("\n--- EXTRAÇÃO CONSOLIDADA CONCLUÍDA ---")
    print(f"Pacotes extraídos: {len(data)}")
    print(f"Entregues: {sum(p['entregue'] for p in data)}")
    print(f"Sem RX: {sum(1 for p in data if not p['entregue'])}")
    for uc in ("UC16", "UC3", "UC1", "UC17"):
        subset = [p for p in data if p["tipo_trafego"] == uc]
        from_log = sum(p["tamanho_origem"] == "UDP_SIZE_LOG" for p in subset)
        fallback = len(subset) - from_log
        print(f"{uc}: {len(subset)} pacotes; tamanho extraído do log={from_log}; payload configurado={fallback}")
    return data


def main():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Parser dos logs Wi-SUN SGHaul")
    parser.add_argument("--excel-linear", default=str(project_root / "data" / "linhas_UDP.xlsx"))
    parser.add_argument("--dir-mesh", default=str(project_root / "LOG__ALL_DLMS" / "51_mesh"))
    parser.add_argument("--dir-quadrada", default=str(project_root / "LOG__ALL_DLMS" / "101_quadrado"))
    parser.add_argument("--output", default=str(project_root / "data" / "dados_lactec_raw.csv"))
    args = parser.parse_args()
    consolidar_logs(args.excel_linear, args.dir_mesh, args.dir_quadrada, args.output)


if __name__ == "__main__":
    main()
