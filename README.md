# SGHaul — códigos corrigidos v2

## Alterações principais

- Horizonte de cada episódio aumentado de **10.000 para 50.000 passos** (`SGHaulEnv(max_steps=50_000)`). Com `step_duration=0.01 s`, isso representa 500 s simulados por episódio.
- Treinamento padrão aumentado de **100.000 para 1.000.000 de transições por modelo**.
- Avaliação via `EvalCallback` a cada **100.000 transições reais**. Como o treinamento usa 4 ambientes vetorizados, o `eval_freq` é convertido para chamadas do VecEnv (`ceil(100000 / 4)`). São usados 3 episódios por avaliação.
- O treinamento só é marcado como concluído após salvar o modelo final (`.done`). A existência isolada de `best_model.zip` não faz o script pular um treino interrompido.
- Diretório de resultados relativo ao projeto, independentemente do diretório atual do PowerShell.
- Buffer mantém contadores incrementais de bytes, evita somar a fila a cada fragmento transmitido e expira pacotes vencidos antes de testar overflow de novas chegadas.
- Recompensa separa violações de deadline e overflow para não penalizar o mesmo overflow duas vezes. Métricas de SLA continuam contabilizando ambos.
- Avaliação retorna `NaN`/N.A. quando não existe pacote no denominador, em vez de declarar SLA de 100% sem amostras.
- Incluída métrica `rho_realized` para comparar a carga efetivamente observada com `rho_target`; a avaliação varre rho de 0.5 até 1.5.

## Estrutura

```text
codigos_corrigidos_v2/
  envs/
    __init__.py
    buffer_manager.py
    sghaul_env.py
    wisun_emulator.py
  log_parser.py
  data_analyzer.py
  train_agent.py
  evaluate_agents.py
  requirements.txt
  data/                 # coloque aqui calibracao.json e dados gerados
```

## Instalação no Windows / PowerShell

Na pasta do projeto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação do ambiente virtual, use diretamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Execução

1. Coloque `calibracao.json` em `data/` (se já tiver sido gerado) e os logs nas pastas esperadas pelo parser.
2. Extraia os logs:

```powershell
py log_parser.py
```

3. Gere/atualize a calibração:

```powershell
py data_analyzer.py
```

4. Treine os modelos:

```powershell
py train_agent.py
```

5. Avalie os modelos:

```powershell
py evaluate_agents.py
```

## Ajustes de custo computacional

Os padrões de treinamento são deliberadamente mais longos. A configuração padrão roda 3 algoritmos × 3 perfis × 3 seeds = 27 modelos, com 1 milhão de transições cada. Para um teste rápido, edite `algorithms`, `profiles` ou `train_seeds` no `train_agent.py`, ou chame `pipeline_somente_treino(max_steps=200_000, episode_max_steps=10_000, n_envs=1)`.

Se houver falta de memória no Windows, reduza `n_envs` de 4 para 2 ou 1. Ao reduzir `n_envs`, o callback continua respeitando aproximadamente o intervalo configurado em transições.

**Importante:** o aumento de passos e de frequência de avaliação melhora a oportunidade de aprendizado e a seleção de checkpoints, mas não garante que SAC/PPO/TD3 superem a heurística. Isso precisa ser confirmado pelos resultados pareados e por `rho_realized`.
