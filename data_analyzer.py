"""Calibra o emulador Wi-SUN a partir do CSV produzido por log_parser.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

UC_LIST = ("UC16", "UC3", "UC1", "UC17")
DEFAULT_PAYLOADS = {"UC16": 100, "UC3": 400, "UC1": 150, "UC17": 150}


def _safe_regression(hops: np.ndarray, latencies: np.ndarray) -> Tuple[float, float, float]:
    valid = np.isfinite(hops) & np.isfinite(latencies)
    x = np.asarray(hops[valid], dtype=float)
    y = np.asarray(latencies[valid], dtype=float)
    if len(x) < 2:
        raise ValueError("menos de dois pares válidos")
    if len(np.unique(x)) < 2:
        slope = 1.0
        intercept = max(0.0, float(np.mean(y) - slope * np.mean(x)))
        predicted = slope * x + intercept
    else:
        fit = linregress(x, y)
        slope = max(1.0, float(fit.slope))
        intercept = max(0.0, float(fit.intercept))
        predicted = slope * x + intercept
    jitter = float(np.std(y - predicted))
    return slope, intercept, jitter


def _estimate_uc3_bursts(times_ms: Iterable[float], burst_gap_ms: float) -> Tuple[float, float, int]:
    """Agrupa timestamps consecutivos; novo burst começa após burst_gap_ms."""
    times = np.sort(np.asarray(list(times_ms), dtype=float))
    times = times[np.isfinite(times)]
    if len(times) == 0:
        return 50.0, 5.0, 0

    # Timestamp repetido representa mensagens simultâneas do mesmo evento.
    unique_times, counts = np.unique(times, return_counts=True)
    starts = [float(unique_times[0])]
    burst_sizes = []
    current_size = int(counts[0])
    for idx in range(1, len(unique_times)):
        gap = float(unique_times[idx] - unique_times[idx - 1])
        if gap > burst_gap_ms:
            burst_sizes.append(current_size)
            starts.append(float(unique_times[idx]))
            current_size = int(counts[idx])
        else:
            current_size += int(counts[idx])
    burst_sizes.append(current_size)

    mean_burst = float(np.mean(burst_sizes))
    if len(starts) >= 2:
        period_seconds = float(np.mean(np.diff(starts)) / 1000.0)
    else:
        period_seconds = 5.0
    return mean_burst, max(period_seconds, 0.001), len(starts)


def _source_groups(frame: pd.DataFrame):
    """Mantém separados arquivos cujos relógios t0 podem reiniciar em zero."""
    if "arquivo_origem" not in frame.columns:
        return [("dataset", frame)]
    source = frame["arquivo_origem"].fillna("dataset").astype(str)
    return [(name, frame.loc[source == name]) for name in source.unique()]


def _total_duration_seconds(frame: pd.DataFrame) -> float:
    durations = []
    for source_name, source_frame in _source_groups(frame):
        times = pd.to_numeric(source_frame["t0"], errors="coerce").dropna()
        if times.empty:
            continue
        duration = float((times.max() - times.min()) / 1000.0)
        if duration <= 0:
            # Arquivo com um único instante: não há como inferir a duração real.
            # 1 s é um fallback explícito, não uma medição.
            print(f"[AVISO] {source_name}: duração do log não inferível; usando 1 s.")
            duration = 1.0
        durations.append(duration)
    return float(sum(durations)) if durations else 1.0


def _estimate_uc3_bursts_by_source(frame: pd.DataFrame, burst_gap_ms: float):
    weighted_burst_sum = 0.0
    total_bursts = 0
    weighted_period_sum = 0.0
    total_intervals = 0
    for _, source_frame in _source_groups(frame):
        mean_burst, period, count = _estimate_uc3_bursts(source_frame["t0"], burst_gap_ms)
        if count > 0:
            weighted_burst_sum += mean_burst * count
            total_bursts += count
        if count > 1:
            weighted_period_sum += period * (count - 1)
            total_intervals += count - 1
    mean_burst = weighted_burst_sum / total_bursts if total_bursts else 0.0
    mean_period = weighted_period_sum / total_intervals if total_intervals else 5.0
    return float(mean_burst), float(max(mean_period, 0.001)), total_bursts


def analisar_dados_lactec(
    caminho_csv: str,
    caminho_json: str,
    burst_gap_ms: float = 1000.0,
) -> Dict[str, dict]:
    csv_path = Path(caminho_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}. Execute log_parser.py primeiro.")
    if burst_gap_ms <= 0:
        raise ValueError("burst_gap_ms deve ser positivo.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("O CSV de entrada está vazio.")

    required = {"topologia", "tipo_trafego", "tamanho_bytes", "hop_count", "latencia_ms", "entregue", "t0"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")

    for col in ("tamanho_bytes", "hop_count", "latencia_ms", "entregue", "t0"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["topologia", "tipo_trafego", "tamanho_bytes", "hop_count", "t0"])
    if df.empty:
        raise ValueError("Não há registros válidos após a limpeza do CSV.")

    calibration: Dict[str, dict] = {}
    for topology in sorted(df["topologia"].astype(str).unique()):
        group = df[df["topologia"].astype(str) == topology].copy()
        params: Dict[str, float] = {}
        beta_values = []
        jitter_values = []

        # Tamanho médio por classe (payload de aplicação; overhead fica no emulador).
        for uc in UC_LIST:
            class_rows = group[group["tipo_trafego"] == uc]
            if not class_rows.empty:
                size = float(class_rows["tamanho_bytes"].median())
                params[f"tamanho_{uc.lower()}"] = max(1.0, size)
            else:
                params[f"tamanho_{uc.lower()}"] = float(DEFAULT_PAYLOADS[uc])
                print(f"[AVISO] {topology}/{uc}: sem amostras; payload default={DEFAULT_PAYLOADS[uc]} B.")

        # Atraso Wi-SUN por salto para cada UC.
        for uc in UC_LIST:
            delivered = group[(group["tipo_trafego"] == uc) & (group["entregue"] == 1)]
            try:
                slope, intercept, jitter = _safe_regression(
                    delivered["hop_count"].to_numpy(), delivered["latencia_ms"].to_numpy()
                )
                params[f"alpha_{uc.lower()}"] = slope
                beta_values.append(intercept)
                jitter_values.append(jitter)
                print(f"[{topology}/{uc}] alpha={slope:.2f} ms/salto; beta={intercept:.2f} ms; jitter={jitter:.2f} ms")
            except ValueError as exc:
                params[f"alpha_{uc.lower()}"] = 20.0
                print(f"[AVISO] {topology}/{uc}: regressão indisponível ({exc}); alpha=20 ms.")

        params["beta"] = float(np.mean(beta_values)) if beta_values else 10.0
        params["std_dev_jitter"] = float(np.mean(jitter_values)) if jitter_values else 5.0

        # PDR por número de saltos.
        pdr_rows = group.dropna(subset=["hop_count", "entregue"])
        pdr_by_hop = pdr_rows.groupby("hop_count")["entregue"].mean()
        if len(pdr_by_hop) >= 2 and pdr_by_hop.index.nunique() >= 2:
            pdr_fit = linregress(pdr_by_hop.index.to_numpy(dtype=float), pdr_by_hop.values.astype(float))
            params["pdr_slope"] = float(pdr_fit.slope)
            params["pdr_intercept"] = float(pdr_fit.intercept)
        elif len(pdr_by_hop) == 1:
            params["pdr_slope"] = 0.0
            params["pdr_intercept"] = float(pdr_by_hop.iloc[0])
        else:
            params["pdr_slope"] = -0.0005
            params["pdr_intercept"] = 1.0014
            print(f"[AVISO] {topology}: PDR sem amostras suficientes; usando fallback.")

        # Duração medida com timestamp TX; evita assumir duração=1 s quando há dados.
        duration_seconds = _total_duration_seconds(group)
        for uc in ("UC16", "UC1", "UC17"):
            class_rows = group[group["tipo_trafego"] == uc]
            params[f"lambda_{uc.lower()}"] = float(len(class_rows) / duration_seconds) if len(class_rows) else 0.0

        uc3_rows = group[group["tipo_trafego"] == "UC3"]
        burst_mean, burst_period, burst_count = _estimate_uc3_bursts_by_source(uc3_rows, burst_gap_ms) if not uc3_rows.empty else (0.0, 5.0, 0)
        if not uc3_rows.empty:
            params["burst_uc3"] = burst_mean
            params["periodo_uc3"] = burst_period
        else:
            # Não inventa uma carga UC3 quando essa classe não aparece nos logs.
            params["burst_uc3"] = 0.0
            params["periodo_uc3"] = 5.0
            print(f"[AVISO] {topology}: sem tráfego UC3 observado; burst_uc3=0.")
        params["uc3_num_bursts_detectados"] = int(burst_count)
        params["uc3_burst_gap_ms"] = float(burst_gap_ms)
        params["duracao_log_segundos"] = float(duration_seconds)
        calibration[topology] = params

    json_path = Path(caminho_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=4, ensure_ascii=False)
    temporary.replace(json_path)
    print(f"\nCalibração salva em: {json_path}")
    return calibration


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Calibração física do emulador Wi-SUN")
    parser.add_argument("--csv", default=str(root / "data" / "dados_lactec_raw.csv"))
    parser.add_argument("--output", default=str(root / "data" / "calibracao.json"))
    parser.add_argument(
        "--burst-gap-ms", type=float, default=1000.0,
        help="Gap máximo entre pacotes que pertencem à mesma rajada UC3 (ms). Ajuste aos logs.",
    )
    args = parser.parse_args()
    analisar_dados_lactec(args.csv, args.output, args.burst_gap_ms)


if __name__ == "__main__":
    main()
