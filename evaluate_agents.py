"""Script Exclusivo de Avaliação e Teste (Fase 2).
Carrega os modelos treinados na Fase 1 (.zip) e avalia contra baselines.
Gera os CSVs finais.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from envs.sghaul_env import SGHaulEnv


class QueueSlackHeuristic:
    def predict(self, observation, deterministic: bool = True):
        obs = np.asarray(observation, dtype=np.float64)
        if obs.ndim > 1:
            obs = obs[0]
        backlog_after = obs[0:3]
        backlog_before = obs[3:6]
        arrivals_bytes = obs[6:9]
        arrivals_packets = obs[9:12]
        min_slack = obs[12:15]
        mean_slack = obs[15:18]
        urgency = np.clip(1.0 - min_slack, 0.0, 2.0)
        average_urgency = np.clip(1.0 - mean_slack, 0.0, 2.0)
        scores = (
            0.25 + 2.0 * backlog_after + 1.5 * backlog_before + 
            1.0 * arrivals_bytes + 0.5 * arrivals_packets + 
            1.5 * urgency + 0.5 * average_urgency
        )
        target = scores / max(float(np.sum(scores)), 1e-12)
        action = _target_allocation_to_action(target)
        return action.astype(np.float32), None


class FixedAllocationPolicy:
    def __init__(self, target_allocation: Sequence[float]):
        target = np.asarray(target_allocation, dtype=float)
        self.target_allocation = target
        self.action = _target_allocation_to_action(target).astype(np.float32)

    def predict(self, observation, deterministic: bool = True):
        batch = np.asarray(observation)
        if batch.ndim > 1:
            return np.repeat(self.action[None, :], batch.shape[0], axis=0), None
        return self.action.copy(), None


def _target_allocation_to_action(target_allocation: Sequence[float]) -> np.ndarray:
    target = np.asarray(target_allocation, dtype=float)
    target = np.maximum(target, 0.020001)
    target = target / target.sum()
    inner = np.maximum((target - 0.02) / 0.94, 1e-6)
    inner = inner / inner.sum()
    logits = np.log(inner)
    logits -= logits.mean()
    return np.clip(logits / 2.0, -1.0, 1.0)


def _safe_sla(violations: int, total: int) -> float:
    # Sem pacotes no denominador, SLA é N/A, não 100% artificial.
    return 100.0 * (1.0 - violations / total) if total > 0 else float("nan")


def evaluate_model_detailed(modelo, env: SGHaulEnv, seeds_list: Iterable[int], rho_targets: Sequence[float]) -> list:
    results = []
    for rho in rho_targets:
        for seed in seeds_list:
            print(f" -> Rodando avaliacao: rho={rho} | seed={seed}", flush=True)
            obs, reset_info = env.reset(seed=int(seed), options={"rho_target": float(rho)})
            topology = reset_info.get("topologia", getattr(env.unwrapped, "topologia_atual", "Unknown"))
            
            keys = ("violacoes_uc16", "violacoes_uc3", "violacoes_mmtc", "total_pacotes_uc16", "total_pacotes_uc3", "total_pacotes_mmtc", "doa_uc16", "doa_uc3", "doa_mmtc", "pdr_drops_uc16", "pdr_drops_uc3", "pdr_drops_mmtc", "generated_uc16", "generated_uc3", "generated_mmtc", "pending_uc16", "pending_uc3", "pending_mmtc", "drops_buffer", "latency_sum_uc16", "latency_sum_uc3", "latency_sum_mmtc", "latency_count_uc16", "latency_count_uc3", "latency_count_mmtc")
            counters = {key: 0 for key in keys}
            throughput, allocations = {"urllc": [], "embb": [], "mmtc": []}, {"urllc": [], "embb": [], "mmtc": []}
            inference_ms = []
            realized_rho = []
            last_info = {}

            for _ in range(env.unwrapped.max_steps):
                start = time.perf_counter()
                action, _ = modelo.predict(obs, deterministic=True)
                inference_ms.append((time.perf_counter() - start) * 1000.0)
                obs, _, terminated, truncated, info = env.step(action)
                last_info = info
                for key in keys:
                    if not key.startswith("pending_"):
                        counters[key] += int(info.get(key, 0))
                realized_rho.append(float(info.get("rho_realized", np.nan)))
                for name in throughput:
                    throughput[name].append(float(info.get(f"throughput_{name}_mbps", 0.0)))
                    allocations[name].append(float(info.get(f"alocado_{name}_mbps", 0.0)))
                if terminated or truncated:
                    break

            pending_url, pending_embb, pending_mmtc = int(last_info.get("pending_uc16", 0)), int(last_info.get("pending_uc3", 0)), int(last_info.get("pending_mmtc", 0))
            sla_url = _safe_sla(counters["violacoes_uc16"] + pending_url, counters["total_pacotes_uc16"])
            sla_embb = _safe_sla(counters["violacoes_uc3"] + pending_embb, counters["total_pacotes_uc3"])
            sla_mmtc = _safe_sla(counters["violacoes_mmtc"] + pending_mmtc, counters["total_pacotes_mmtc"])
            e2e_url = _safe_sla(counters["pdr_drops_uc16"] + counters["doa_uc16"] + counters["violacoes_uc16"] + pending_url, counters["generated_uc16"])
            e2e_embb = _safe_sla(counters["pdr_drops_uc3"] + counters["doa_uc3"] + counters["violacoes_uc3"] + pending_embb, counters["generated_uc3"])
            e2e_mmtc = _safe_sla(counters["pdr_drops_mmtc"] + counters["doa_mmtc"] + counters["violacoes_mmtc"] + pending_mmtc, counters["generated_mmtc"])

            record = {
                "eval_seed": int(seed), "rho_target": float(rho), "rho_realized": float(np.nanmean(realized_rho)) if realized_rho and np.isfinite(realized_rho).any() else float("nan"), "topologia": topology,
                "sla_u": sla_url, "sla_e": sla_embb, "sla_m": sla_mmtc,
                "reliability_e2e_u": e2e_url, "reliability_e2e_e": e2e_embb, "reliability_e2e_m": e2e_mmtc,
                "lat_u": counters["latency_sum_uc16"] / max(1, counters["latency_count_uc16"]),
                "lat_e": counters["latency_sum_uc3"] / max(1, counters["latency_count_uc3"]),
                "lat_m": counters["latency_sum_mmtc"] / max(1, counters["latency_count_mmtc"]),
                "inferencia_ms": float(np.mean(inference_ms)) if inference_ms else 0.0,
                "thp_urllc": float(np.mean(throughput["urllc"])) if throughput["urllc"] else 0.0,
                "thp_embb": float(np.mean(throughput["embb"])) if throughput["embb"] else 0.0,
                "thp_mmtc": float(np.mean(throughput["mmtc"])) if throughput["mmtc"] else 0.0,
                "aloc_urllc": float(np.mean(allocations["urllc"])) if allocations["urllc"] else 0.0,
                "aloc_embb": float(np.mean(allocations["embb"])) if allocations["embb"] else 0.0,
                "aloc_mmtc": float(np.mean(allocations["mmtc"])) if allocations["mmtc"] else 0.0,
                "doa_urllc": counters["doa_uc16"], "doa_embb": counters["doa_uc3"], "doa_mmtc": counters["doa_mmtc"],
                "pdr_drops_total": counters["pdr_drops_uc16"] + counters["pdr_drops_uc3"] + counters["pdr_drops_mmtc"],
                "pending_total": pending_url + pending_embb + pending_mmtc
            }
            results.append(record)
    return results


def pipeline_somente_avaliacao():
    project_root = Path(__file__).resolve().parent
    output_root = project_root / "resultados"
    data_dir = output_root / "data"
    model_dir = output_root / "models" / "final_pipeline"
    data_dir.mkdir(parents=True, exist_ok=True)

    #test_seeds = list(range(201, 221))
    test_seeds = list(range(201, 202))
    # Varredura de carga: inclui subcarga, próximo da saturação e sobrecarga.
    #rho_levels = [0.5, 0.8, 0.95, 1.1, 1.3, 1.5]
    rho_levels = [0.95, 1.1]
    
    profiles = ["Balanceado", "Foco_eMBB", "Missao_Critica"]
    algorithms = ["SAC"]#, "PPO", "TD3"]
    train_seeds = [10, 20, 30]
    
    test_rows = []
    algorithm_classes = {"SAC": SAC, "PPO": PPO, "TD3": TD3}
    
    # 1. Avaliar as Heurísticas e Baselines Estáticos primeiro (Roda Apenas 1 Vez)
    baseline_policies = {
        "Heuristica_Backlog_Slack": QueueSlackHeuristic(),
        "Estatico_33_33_33": FixedAllocationPolicy((1 / 3, 1 / 3, 1 / 3)),
    }
    
    print("=== INICIANDO AVALIAÇÃO DE BASELINES ===")
    env = SGHaulEnv()
    for policy_name, policy in baseline_policies.items():
        print(f"\nAvaliando Baseline: {policy_name}")
        rows = evaluate_model_detailed(policy, env, test_seeds, rho_levels)
        for row in rows:
            row.update({"modelo": policy_name, "perfil": "N/A", "train_seed": 0})
        test_rows.extend(rows)

    # 2. Carregar os modelos treinados e testar
    print("\n=== INICIANDO AVALIAÇÃO DOS AGENTES TREINADOS ===")
    for algo in algorithms:
        for profile in profiles:
            for train_seed in train_seeds:
                stem = f"{algo}_{profile}_seed-{train_seed}"
                
                # Procura a melhor versão, ou usa a última se best não existir
                best_path = model_dir / "best" / stem / "best_model.zip"
                final_path = model_dir / f"{stem}.zip"
                model_path = best_path if best_path.exists() else final_path
                
                if not model_path.exists():
                    print(f"[AVISO] Checkpoint ignorado (não encontrado): {stem}")
                    continue
                    
                print(f"\nAvaliando Modelo: {algo} | Perfil: {profile} | Seed Treino: {train_seed}")
                best_model = algorithm_classes[algo].load(str(model_path))
                
                rows = evaluate_model_detailed(best_model, env, test_seeds, rho_levels)
                for row in rows:
                    row.update({
                        "modelo": algo,
                        "perfil": profile,
                        "train_seed": train_seed,
                    })
                test_rows.extend(rows)
                del best_model
                gc.collect()

    # 3. Exportar resultados E2E e Sumário
    test_df = pd.DataFrame(test_rows)
    test_csv = data_dir / "test_results_paired_final.csv"
    test_df.to_csv(test_csv, index=False)
    
    summary = (
        test_df.groupby(["modelo", "perfil", "rho_target"])
        .agg({"sla_u": "mean", "sla_e": "mean", "reliability_e2e_u": "mean", "pending_total": "mean"})
        .reset_index()
    )
    summary_csv = data_dir / "test_summary_final.csv"
    summary.to_csv(summary_csv, index=False)
    
    print("\n[FIM] Todos os resultados avaliados. CSVs gerados na pasta 'resultados/data'")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    pipeline_somente_avaliacao()
