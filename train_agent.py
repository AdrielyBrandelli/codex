"""Script Exclusivo de Treinamento (Fase 1).
Apenas roda o treinamento dos agentes e salva os checkpoints na pasta.
NÃO realiza avaliações detalhadas nem roda heurísticas.
"""
from __future__ import annotations
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
import gc
import os
import time
import torch
torch.set_num_threads(1)  # Limita as threads para não estourar a CPU e a RAM
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.sghaul_env import SGHaulEnv


def _build_model(algo: str, env, seed: int):
    if algo == "SAC":
        return SAC("MlpPolicy", env, verbose=0, seed=seed, learning_rate=3e-4)
    if algo == "PPO":
        return PPO("MlpPolicy", env, verbose=0, seed=seed, learning_rate=3e-4)
    if algo == "TD3":
        noise = NormalActionNoise(mean=np.zeros(3), sigma=0.1 * np.ones(3))
        return TD3("MlpPolicy", env, action_noise=noise, verbose=0, seed=seed, learning_rate=3e-4)
    raise ValueError(f"Algoritmo desconhecido: {algo}")


def pipeline_somente_treino(
    #max_steps: int = 1_000_000,
    max_steps: int = 1_000_000,
    results_dir: Optional[str] = "resultados",
    train_rho: float = 0.95,
    episode_max_steps: int = 50_000,
    #episode_max_steps: int = 20_000,
    eval_interval_transitions: int = 100_000,
    #eval_interval_transitions: int = 80_000,
    n_envs: int = 4,
    #n_eval_episodes: int = 3,
    n_eval_episodes: int = 3,
):
    project_root = Path(__file__).resolve().parent
    output_root = (project_root / results_dir).resolve() if results_dir else project_root / "resultados"
    model_dir = output_root / "models" / "final_pipeline"
    model_dir.mkdir(parents=True, exist_ok=True)

    profiles = [
        {"nome": "Balanceado", "w1": 4.0, "w2": 3.0, "w3": 2.0, "w4": 2.0},
        {"nome": "Foco_eMBB", "w1": 6.0, "w2": 2.0, "w3": 2.0, "w4": 2.0},
        {"nome": "Missao_Critica", "w1": 2.0, "w2": 6.0, "w3": 2.0, "w4": 3.0},
    ]
    
    # ATENÇÃO: Para rodar um de cada vez em dias diferentes, deixe apenas um aqui:
    # algorithms = ["TD3"]
    algorithms = ["SAC"]#, "PPO"]
    train_seeds = [10, 20, 30]
    #train_seeds = [10]

    for algo in algorithms:
        for profile in profiles:
            for train_seed in train_seeds:
                stem = f"{algo}_{profile['nome']}_seed-{train_seed}"
                final_model_path = model_dir / f"{stem}.zip"
                best_dir = model_dir / "best" / stem
                best_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = best_dir / "best_model.zip"
                completion_marker = model_dir / f"{stem}.done"

                if completion_marker.exists() and final_model_path.exists():
                    print(f"[SKIP] {stem}: treinamento concluído anteriormente.")
                    continue

                env_kwargs = {
                    "w1": profile["w1"], "w2": profile["w2"],
                    "w3": profile["w3"], "w4": profile["w4"],
                    "rho_target": train_rho,
                }
                
                # Se der WinError de paginação novamente, reduza n_envs para 2 ou 1
                train_env = make_vec_env(
                    SGHaulEnv, n_envs=n_envs, env_kwargs={**env_kwargs, "max_steps": episode_max_steps},
                    vec_env_cls=SubprocVecEnv, seed=train_seed,
                )
                eval_env = Monitor(SGHaulEnv(**env_kwargs, max_steps=episode_max_steps, calibration_path="data/calibracao.json"))
                eval_env.reset(seed=train_seed + 100_000)
                model = _build_model(algo, train_env, train_seed)

                # SB3 conta chamadas ao VecEnv; cada chamada avança n_envs transições.
                eval_freq_calls = max(1, int(np.ceil(eval_interval_transitions / n_envs)))
                # NOVO: Para o treino se não houver melhoria durante 3 avaliações seguidas
                stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=4, verbose=1)
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=str(best_dir),
                    log_path=str(best_dir / "eval_logs"),
                    eval_freq=eval_freq_calls,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    verbose=1,
                    callback_after_eval=stop_train_callback  # <-- Adicionado aqui
                )

                print(
                    f"\nTreinando {algo} | perfil={profile['nome']} | seed={train_seed} "
                    f"| transições={max_steps:,} | horizonte={episode_max_steps:,} "
                    f"| avaliação a cada {eval_interval_transitions:,} transições"
                )
                model.learn(total_timesteps=max_steps, callback=eval_callback, progress_bar=True)
                #model.learn(total_timesteps=max_steps, progress_bar=True)
                model.save(str(final_model_path))
                completion_marker.write_text(
                    f"algorithm={algo}\nprofile={profile['nome']}\nseed={train_seed}\n"
                    f"total_timesteps={max_steps}\nepisode_max_steps={episode_max_steps}\n",
                    encoding="utf-8",
                )

                del model, train_env, eval_env
                gc.collect()

    print("\n[FIM] Todos os treinamentos configurados foram concluídos e salvos.")

if __name__ == "__main__":
    pipeline_somente_treino()
