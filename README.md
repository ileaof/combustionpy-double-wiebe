# Double Wiebe Combustion Analysis

Simulação e calibração de um modelo de combustão **Double Wiebe** (duas
funções de Wiebe ponderadas por *alpha*) acoplado a um modelo termodinâmico
de zona única com transferência de calor de Hohenberg, para motores de
combustão interna de pistão.

> *This combustion simulation employs a double Wiebe function and extends the
> single Wiebe model developed as part of L. Queiroz's M.Sc. thesis under the
> supervision of Prof. I. L. Ferreira.*

Extensão do projeto Single Wiebe: <https://github.com/ileaof/combustionpy-single-wiebe>

---

## 1. Instalação

Requisitos: **Python 3.11+** e pip.

```bash
cd double_wiebe
pip install -e .            # instala o pacote + o comando `double-wiebe`
# alternativa:
pip install -r requirements.txt
```

Verificação rápida:

```bash
double-wiebe --help
```

## 2. Como iniciar uma análise

### 2.1 Dados experimentais

Arquivo texto (`.txt`/`.csv`/`.tsv`) com **duas colunas numéricas**:
ângulo do virabrequim e pressão no cilindro. Unidades aceitas:
ângulo em **radianos ou graus**; pressão em **Pa, kPa ou bar**.
Exemplo incluído: `data/example_pressure.txt` (θ em rad, P em bar).

### 2.2 Arquivo de configuração

```bash
double-wiebe example-config configs/meu_config.yaml
```

O YAML tem as seções `data`, `engine`, `wiebe`, `simulation` e
`calibration` (ângulos em **graus** no YAML — convertidos para radianos
internamente). Veja `configs/example.yaml` comentado.

### 2.3 Simulação (CLI)

```bash
double-wiebe simulate --data data/example_pressure.txt \
    --config configs/example.yaml --output outputs/run_01
```

Sobrescritas rápidas (sem editar o YAML), repetíveis:

```bash
double-wiebe simulate --data data/exemplo.txt \
    --set wiebe.alpha=0.4 --set engine.Rc=17.0 --output outputs/run_02
```

### 2.4 Calibração (CLI)

```bash
double-wiebe calibrate --data data/example_pressure.txt \
    --config configs/example.yaml \
    --method differential-evolution --seed 42 \
    --select Rc,theta01,delta1,m1,theta02,delta2,m2,alpha \
    --output outputs/calib_01
```

Métodos: `differential-evolution` (global, scipy), `pso` (opcional) e
`least-squares` (refinamento local; também usado no *polish* final).

**Recomendação**: não calibre os 10 parâmetros de uma vez. Comece com 2–4
parâmetros fisicamente identificados (por ex. `theta01, delta1, alpha`) e
observe os alertas de *sensibilidade/bordas* impressos ao final — parâmetros
"insensíveis" indicam identificabilidade fraca.

### 2.5 Outros comandos

| Comando | Função |
|---|---|
| `double-wiebe validate --data ... --config ...` | valida dados/config sem simular |
| `double-wiebe plot outputs/run_01 --out outputs/figs` | regenera gráficos de um resultado |
| `double-wiebe gui` | abre a interface gráfica (Streamlit) |
| `double-wiebe example-config [arquivo]` | imprime/grava YAML de exemplo |

Códigos de saída: **0** ok · **1** falha de execução · **2** entrada inválida.

### 2.6 Interface gráfica

```bash
double-wiebe gui            # ou: streamlit run src/double_wiebe/gui.py
```

Aba *Experimental Data* → carregue o arquivo, confira unidades e intervalo →
aba *Engine Setup* → aba *Double Wiebe* (prévia instantânea das fases) →
aba *Simulation* → **Run simulation** → aba *Results* (9 gráficos) →
aba *Export* (downloads). A calibração roda em thread com progresso e
cancelamento, e o botão "Aplicar parâmetros calibrados" transfere o
resultado para a simulação. **CLI e GUI usam exatamente o mesmo núcleo
científico** — não há duplicação de equações.

## 3. Modelo

### 3.1 Double Wiebe

Duas fases (1 = pré-misturada, 2 = difusão), cada uma com início θ0, duração
δ, fator de forma m e eficiência a:

```
z_j = (θ − θ0_j)/δ_j        (z_j < 0 ⇒ x_j = 0)
x_j = 1 − exp(−a_j · z_j^(m_j+1))
dx_j/dθ = a_j (m_j+1)/δ_j · z_j^m_j · exp(−a_j · z_j^(m_j+1))
x_b  = α·x_1 + (1−α)·x_2
```

Modos:
* **continuous** — cauda assintótica após θ0+δ (igual ao Single Wiebe);
* **bounded** — após θ0+δ a fase satura em 1−exp(−a_j) com derivada nula.

### 3.2 Termodinâmica de zona única (3 EDOs)

Estado `Y = [P (kPa), T_g (K), Q_parede (J)]`, integrado por
`solve_ivp` (DOP853/LSODA/RK45, rtol/atol 1e-9) nos ângulos experimentais:

```
dP/dθ  = 1/V [ (κ−1)(dQ/dθ − dQ_w/dθ) − κ P dV/dθ ]      (kPa/rad)
dT_g/dθ = (T1/(P1 V1)) (V dP/dθ + P dV/dθ)
dQ_w/dθ = h A_s (T_g − T_w) / ω                          (J/rad)
```

com `dQ/dθ = Q_total · dx_b/dθ`, `Q_total = m_comb · PCI` (kJ/ciclo),
geometria biela-manivela exata e **Hohenberg**:

```
h = 130 V^(−0.06) (P·10⁻²)^0.8 T_g^(−0.4) (V_p + 1.4)^0.8   [W/m²K]
```

### 3.3 Valores de referência (motor de exemplo)

86 mm de diâmetro · 70 mm de curso · biela 117,5 mm · 3396,20 rpm ·
Rc 17 · m_comb = 9,42754647351e−6 kg/ciclo · PCI 39.191,3 kJ/kg ·
κ 1,37 · T1 308,15 K · T_parede 440 K.

## 4. Calibração

* **Parâmetros (10):** Rc, theta01, delta1, m1, a1, theta02, delta2, m2,
  a2, alpha (ordem canônica em `PARAM_ORDER`).
* **Objetivo:** RMSE (P_exp − P_sim) + penalidades de regularização:
  ordem das fases (θ02 ≥ θ01), sobreposição, durações mínimas (δ_min),
  *ridge* opcional.
* **Fluxo:** busca global (DE/PSO, com semente) → *polish* least-squares
  (resíduo completo P_sim−P_exp) → sensibilidade ±1% → alertas de
  identificabilidade e de bordas do domínio.
* **Reprodutibilidade:** `--seed` fixa a semente (DE e PSO).

## 5. Unidades

| Grandeza | Interface (CLI/GUI/YAML) | Interno | Exportações |
|---|---|---|---|
| Ângulo | radianos ou graus | rad | rad + graus (results.csv) |
| Pressão | Pa, kPa, bar | kPa | kPa |
| Volume | m³ | m³ | m³ |
| Temperatura | K | K | K |
| Calor liberado | kJ | kJ | kJ (taxas em kJ/rad) |
| Calor à parede | J (acumulado) | J | J |
| rpm | rpm | — | rpm |

## 6. Exportações (por execução)

`results.csv` (14 colunas), `parameters.yaml`, `metrics.json`,
`convergence.csv` (quando há calibração), `pressure_comparison.png`/`.pdf`,
`heat_release.png`, `burned_fraction.png`, `report.html` (autocontido,
imagens embutidas).

## 7. Testes

```bash
python -m pytest tests -q
```

Cobrem: Wiebe (zero antes do início, monotonia, derivada analítica,
modos, redução ao Single Wiebe), geometria (volume/derivada/área),
simulação (sem NaN, indicadores, penalidade), calibração (reprodutibilidade,
limites, regularização), dados (unidades, filtros, erros) e CLI
(códigos de saída, `--set`, proteção de diretório, equivalência CLI×núcleo).

## 8. Estrutura

```
double_wiebe/
├── pyproject.toml
├── requirements.txt
├── configs/example.yaml
├── data/example_pressure.txt
├── outputs/                     # resultados (não versionados)
├── src/double_wiebe/
│   ├── __init__.py   models.py  wiebe.py  geometry.py
│   ├── thermodynamics.py  simulation.py  calibration.py
│   ├── data_processing.py metrics.py  plotting.py
│   ├── reporting.py   cli.py    gui.py
└── tests/
```

## 9. Limitações físicas e numéricas

1. **Zona única** — sem diferenças espaciais; T_g é média da câmara.
2. **κ e composição constantes** — sem variação de calores específicos com
   temperatura/composição (sem equilíbrio químico, sem dissociação).
3. **Sem blow-by** — vazamento pelos anéis não modelado.
4. **Hohenberg empírica** — constantes genéricas (130, 1.4); calibrar para
   o motor real se necessário.
5. **P1 = P(IVC) da própria série** — a simulação começa no primeiro ponto
   experimental; erros de transdutor perto de IVC propagam-se.
6. **Wiebe é paramétrico** — descreve, não explica: parâmetros correlacionados
   podem reproduzir a mesma pressão (ver alertas de sensibilidade).
7. **Modo bounded tem um pequeno degrau** em θ0+δ (derivada descontínua).
8. **Calibração global com muitos parâmetros livres** é mal condicionada —
   use poucos parâmetros e inspecione os alertas.
9. **Ângulos** relativos ao PMS (θ = 0); ignição/IGV não são modeladas
   explicitamente (θ0_j é o início da combustão).

## 10. Sugestões de validação contra o Single Wiebe

1. Simular o mesmo experimento com α = 1 e as fases idênticas — o Double
   Wiebe reduz-se ao Single Wiebe (`tests/test_wiebe.py` verifica
   analiticamente) e os resultados devem coincidir.
2. Comparar RMSE/R² Single vs Double no mesmo conjunto: o Double Wiebe
   deve ganhar em ciclos com combustão bifásica (ex.: diesel/CI) e
   empatar no monofásico (SI).
3. Inspecionar as taxas dQ₁/dθ e dQ₂/dθ: a fase pré-misturada deve
   corresponder ao pico inicial de liberação de calor e a fase de difusão
   à cauda longa.
4. Verificar conservação: `E1 + E2 ≈ Q_total·(média ponderada das frações
   finais)` e calor perdido plausível (10–25 % do liberado).
5. Validar parâmetros calibrados contra literatura (δ1 ~ 20–40°, θ01 perto
   do ponto de ignição) e contra os indicadores experimentais (ângulo do
   pico de pressão, P_max).