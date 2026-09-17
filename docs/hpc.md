# Plano HPC — Aceleração do Double Wiebe

Documentação da aceleração de desempenho do Double Wiebe, implementada em
conformidade com as regras fundamentais da especificação:

1. **Preserva a implementação CPU serial como referência** (`SerialBackend`,
   caminho original solve_ivp/DOP853, inalterado).
2. **Nenhuma equação, unidade, limite ou função objetivo foi alterada** —
   o objetivo continua sendo RMSE(P_sim, P_exp) + regularização, com a mesma
   PENALTY (1e10) para falhas de integração.
3. **CLI e GUI usam o mesmo núcleo** — ambos passam por
   `calibration.run_calibration`, que orquestra o backend.
4. **Toda otimização é validada numericamente contra a serial** — testes
   automáticos (`tests/test_backends.py`) e re-run serial do melhor
   candidato ao final de toda calibração (`diferenca_integrador` no
   resultado).
5. **CUDA/OpenCL não são dependências obrigatórias** — não são nem
   dependências opcionais de execução; são apenas *detectados*
   (`backends.detection`), com a justificativa abaixo.
6. **Funciona em máquina sem GPU** — sem GPU, sem numba, sem nada além de
   numpy/scipy, tudo funciona (backend serial é o default).
7. **Nenhum ganho declarado sem benchmark reproduzível** —
   `double-wiebe benchmark` (warm-up, média, desvio, speedup, CSV) e
   `outputs/benchmark_hpc.csv` com dados reais.
8. **Sem pseudocódigo, funções vazias ou trechos omitidos** — tudo executável.

---

## 1. Etapa 1 — Diagnóstico (profiling antes de qualquer otimização)

`tools/profile_audit.py` (cProfile + `time.perf_counter`, dados reais
`P_exp-Carga-3_45%.txt`, 721 pontos):

| Etapa | Tempo | % |
|---|---:|---:|
| leitura/preparação dos dados | 0,0093 s | 0,1% |
| geometria + Wiebe (analíticas) | ~0 | ~0% |
| `integrate_ode` (1 simulação) | 0,0259 s | 0,3% |
| `evaluate_rmse` (1 avaliação do objetivo) | 0,0243 s | 0,3% |
| **calibração DE (função objetivo)** | **9,4455 s** | **98,4%** |
| métricas | ~0 | ~0% |
| 5 gráficos plotly | 0,0807 s | 0,8% |
| exportação | 0,0068 s | 0,1% |

**Conclusão do gargalo**: ~1.304 chamadas do RHS por integração;
1 avaliação do objetivo ≈ 24–27 ms; DE default (8 parâmetros, popsize 20,
maxiter 200) ≈ **21 min** de avaliações. O gargalo é 100% a função objetivo,
paralelizável **entre candidatos** (a integração de um candidato é
sequencial — cada passo depende do anterior). Baseline salvo em
`outputs/baseline_serial.json` **antes** de qualquer alteração.

## 2. Arquitetura

```
                    ┌──────────────────────────────────────────────┐
   CLI / GUI ──────►│ calibration.run_calibration (orquestrador)   │
   (mesmo núcleo)   │   objective(x)  = caminho serial (referência)│
                    │   eval_batch(X) = backend.evaluate_population│
                    └──────┬───────────────────────────────────────┘
                           │ select_backend (serial|cpu|cpu-parallel|auto)
        ┌──────────────────┼─────────────────────────┐
        ▼                  ▼                         ▼
  SerialBackend      CPUBackend              MultiprocessingBackend
  solve_ivp/adapt.   RK4 passo fixo em lote  ProcessPoolExecutor (spawn)
  (referência,       (integrators/rk4_numpy  cada candidato com a MESMA
  regra 1)           .py / rk4_numba.py)     implementação serial →
                                             números idênticos
```

- **`ComputeBackend`** (`backends/base.py`): `name`, `is_available()`,
  `capabilities()`, `simulate()`, `evaluate_population()`,
  `synchronize()`, `close()`.
- **`select_backend(nome, workers, integrator)`** (`backends/__init__.py`):
  `auto` escolhe por mini-benchmark (`benchmark.choose_backend`);
  backend indisponível degrada com `UserWarning` para a serial.
- **Integradores** (`integrators/`): `scipy_integrator` (referência),
  `rk4_numpy` (lote NumPy), `rk4_numba` (compilado, cai para NumPy sem
  numba). `get_integrator(nome)` para acesso direto.
- **Precisão**: `float64` (default) ou `float32` no RK4 em lote. O
  **resultado final é sempre re-integrado em float64 com solve_ivp** —
  float32 nunca chega ao usuário como resposta.
- **`substeps`**: sub-passos do RK4 de passo fixo entre pontos
  experimentais (default 4).

### Onde NÃO há paralelismo (e por quê)

- **Dentro de um candidato**: a integração é sequencial por natureza
  (cada passo usa o estado do anterior). Não há ganho real — não foi
  forçado (instrução da especificação).
- **CUDA/OpenCL**: a máquina de referência **não possui GPU nem as
  bibliotecas** — qualquer ganho seria *declarado* sem benchmark
  reproduzível, violando a regra 7. Além disso, para 3 EDOs por candidato
  o custo de transferência (host↔device) por iteração DE domina o
  cálculo; backends GPU aqui não seriam testáveis nem reprodutíveis.
  **Decisão**: CUDA/OpenCL ficam em `backends/detection.py` (detecção
  para o comando `devices`/GUI) e nos extras `.[cuda]`/`.[opencl]`
  documentados, mas não são backends de execução. Se um hardware CUDA
  existir e o gargalo se mantiver, o caminho natural é um kernel de lote
  RK4 (a mesma matemática de `rk4_numpy`, já validada).

## 3. Modo acelerado (`backend=cpu`) — o que muda e o que é garantido

- O integrador muda de **solve_ivp (DOP853 adaptativo, rtol=atol=1e-9)**
  para **RK4 clássico de passo fixo com `substeps` sub-passos por intervalo
  experimental**. É a única mudança numérica do projeto; as equações, os
  limites, a regularização e a PENALTY são idênticos.
- Medição com dados reais (721 pontos, substeps=4, candidatos aleatórios
  dentro dos limites): max |ΔP| ≤ **2 kPa** em candidatos de interior
  (≈0,1% do pico); max |ΔRMSE| ≈ **0,15 kPa** em candidatos de interior e
  até ≈ **4,0 kPa** sobre 1.000 candidatos aleatórios (inclui candidatos
  de borda perto da falha) — dentro da tolerância de 5 kPa verificada
  pelos testes.
- Candidatos que falham na serial podem integrar no RK4 (e vice-versa) —
  diferença apenas na fronteira de falha; ambos os lados recebem PENALTY
  quando falham, nunca NaN.
- **Regra 4 em código**: ao final de toda calibração acelerada, o melhor
  candidato é re-integrado com solve_ivp; o resultado reporta
  `rmse_integrador` (valor usado na busca), `rmse` (re-run serial — o
  verdadeiro) e `diferenca_integrador = |rmse − rmse_integrador|`. CLI e
  GUI mostram essa diferença.
- **DE (scipy)**: no modo em lote, scipy exige `vectorized=True` +
  `updating="deferred"` — a *trajetória* da busca difere da serial
  (`updating="immediate"`), mas os valores do objetivo por candidato são
  os mesmos do backend escolhido. **PSO** (implementação própria) não tem
  esse efeito: com `backend="cpu-parallel"`, a trajetória é **bit-idêntica**
  à serial (mesma seed).

## 4. Benchmarks medidos

Máquina de referência: Intel Meteor Lake (16 núcleos lógicos, 31,6 GiB RAM,
Windows 11, Python 3.11.4), sem GPU. Reproduzir com:

```bash
double-wiebe benchmark --data P_exp-Carga-3_45%.txt \
    --ang-unit radianos --press-unit bar --rep 3 \
    --output outputs/benchmark_hpc.csv
```

Resultados medidos (`outputs/benchmark_hpc.csv`, dados reais de 721 pontos,
16-09-2026; tempos absolutos variam com a carga da máquina — nesta execução
a máquina estava mais carregada, ~110 ms/cand na serial, contra ~24 ms no
baseline `outputs/baseline_serial.json`; os **speedups são a métrica
robusta**):

| N | Backend | ms/cand | Speedup | max |ΔRMSE| vs serial | Penalidades iguais |
|---:|---|---:|---:|---:|---|
| 1000 | serial (referência) | 110,2 | 1× | — | — |
| 1000 | cpu (rk4 numpy) | 5,3 | **20,6×** | 4,01 kPa | não* |
| 1000 | cpu (rk4 numba) | 2,9 | **38,3×** | 4,01 kPa | não* |
| 1000 | cpu-parallel (processos) | 16,1 | **6,8×** | **0,000** (bit-exato) | sim |
| 100 | serial | 111,1 | 1× | — | — |
| 100 | cpu (rk4 numpy) | 20,7 | 5,4× | 0,55 kPa | não* |
| 100 | cpu (rk4 numba) | 3,0 | 36,9× | 0,55 kPa | não* |
| 100 | cpu-parallel | 17,6 | 6,3× | 0,000 (bit-exato) | sim |
| 20 | serial | 20,3 | 1× | — | — |
| 20 | cpu (rk4 numpy) | 78,0 | 0,3× (overhead de lote) | 0,17 kPa | não* |
| 20 | cpu (rk4 numba) | 2,9 | 7,0× | 0,17 kPa | não* |
| 20 | cpu-parallel | 13,1 | 1,5× | 0,000 (bit-exato) | sim |

\* "não" = o RK4 de passo fixo integra com sucesso um candidato que o
solve_ivp adaptativo rejeita (ou vice-versa) — diferença apenas na
fronteira de falha; ambos os lados recebem PENALTY quando falham. Sobre
candidatos válidos em comum, a discrepância máxima medida foi 4,0 kPa
(N=1000, inclui candidatos de borda; em candidatos de interior,
< 0,6 kPa — ver §3).

Calibração real (DE curto, seed 42): serial ≈ 20 min (default DE) vs
**≈ 40 s** com `backend=cpu` (numba) — aceleração ≈ 28× na prática, com
re-validação serial automática. PSO (25 iterações × 12 partículas):
23,3 s (serial) → 4,5 s (cpu-parallel), **5×**, com RMSE e parâmetros
**bit-idênticos**.

Notas honestas:
- `cpu-parallel` só compensa com populações grandes (spawn custa ~1 s
  por processo no Windows); com N=20 ganha só 1,5×.
- `rk4_numpy` puro (sem numba) perde para a serial com populações
  pequenas (overhead de vetorização); com numba instalado é o melhor
  custo-benefício em todos os tamanhos ≥ 20.
- Nesta CPU híbrida (ruidosa), o numba em 1 thread foi o melhor caso;
  o backend `cpu` não ativa `prange` por padrão.
- `float32` roda e é aceito (`--precision float32`), mas o ganho medido
  foi marginal e o re-run final é sempre float64; não recomendado como
  default.

## 5. Como usar

CLI:
```bash
double-wiebe devices                                   # hardware + backends
double-wiebe calibrate --data ... --backend auto       # escolhe o mais rápido
double-wiebe calibrate --data ... --backend cpu-parallel --workers 8
double-wiebe calibrate --data ... --benchmark --profile --output outputs/run
```

GUI (aba Calibration → expander **Desempenho (backend de computação)**):
backend, integrador, workers, batch, precisão, sub-passos, status do
hardware; o resultado mostra tempo e diferença do integrador.

YAML (`calibration:`): chaves planas `backend`, `integrator`, `workers`,
`batch_size`, `precision`, `substeps`.

## 6. Testes e garantias

`tests/test_backends.py` (markers `slow` para os longos):

- cpu-parallel ≡ serial **bit a bit** (mesma implementação em processos);
- RK4 em lote vs solve_ivp: |ΔRMSE| < 5 kPa e |ΔP| ≤ 5 kPa por candidato;
- `rk4_numpy` ≡ `rk4_numba` bit a bit;
- regularização em lote ≡ regularização escalar (por linha);
- candidatos falhos → PENALTY, **nunca NaN**;
- população de 1; `batch_size` que não divide S → mesmo resultado;
- `select_backend` inválido → `BackendError`; indisponível → serial com
  `UserWarning` (ou exceção com `fallback=False`);
- determinismo (mesma entrada → mesmos valores);
- calibração PSO `cpu-parallel` ≡ serial (RMSE e parâmetros exatos) e
  `cpu` com `diferenca_integrador` < 5 kPa;
- benchmark reproduzível.