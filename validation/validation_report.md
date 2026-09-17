# Relatório de Validação — Paridade Mathematica × Python (Double Wiebe)

**Data**: 2026-09-17
**Escopo**: verificar se a implementação Python (`double_wiebe`) reproduz o código e os
resultados do notebook Mathematica `Modelo_Double_Wiebe_v2.nb` (MOD0d).
**Fontes técnicas**: `Modelo_Double_Wiebe_v2.nb` (1,09 MB, lido no formato bruto para os
pontos críticos) e `modelo-double-wiebe.pdf` (15 páginas, renderização do notebook).
Os documentos foram tratados **exclusivamente como fontes técnicas**.

---

## 1. Resumo Executivo

A implementação Python do modelo Double Wiebe **reproduz o MOD0d do notebook**:

| Critério | Âncora Mathematica | Resultado Python | \|Δ\| | Tolerância | Situação |
|---|---:|---:|---:|---:|---|
| Caso obrigatório (etapa 10) | 100.538 kPa | **100.537707 kPa** | 0.000293 | ±0.5 kPa | ✔ aprovado |
| Vetor ativo do PSO, k=190 (etapa 11) | 55.381 kPa | **55.380933 kPa** | 0.000067 | ±0.5 kPa | ✔ aprovado |
| Ponto a ponto (459 pontos, caso A) | — | max \|ΔP\| = 7.5e-4 kPa | 7.5e-4 | ≤1.0 kPa | ✔ aprovado |
| Ponto a ponto (caso A) | — | max \|ΔTg\| = 1.4e-4 K | 1.4e-4 | ≤1.0 K | ✔ aprovado |

A função de erro do notebook `erro = sqrt(SSres/(q−2))` foi **preservada** (q−2.0, divisor
nunca alterado) e reproduz as âncoras com margem ~1700× dentro da tolerância.

**Classificação final: VALIDADO COM RESSALVAS** — as ressalvas (§10) são: ausência de
Wolfram Engine no ambiente (a comparação ponto a ponto usa re-implementação independente
das equações extraídas, ancorada nos resultados do próprio Mathematica); o resultado
comentado k=532 (34.2586 kPa) é uma saída histórica inconsistente com o MOD0d ativo;
`metrics.rmse` do pacote usa divisor `q` (divergência documentada, comparada em coluna
própria); os limites de calibração do Python são um superset intencional dos do notebook.

---

## 2. Metodologia

1. **Extração das equações**: o PDF foi integralmente lido (renderização das células
   ativas do notebook); o `.nb` bruto foi consultado nos pontos que o PDF não resolve
   com certeza: o divisor da função de erro (confirmado `Sqrt[SSres/(q−2.)]` via
   `SqrtBox` no formato bruto) e o resultado comentado k=532 (confirmado na
   `InterpretationBox` da célula comentada).
2. **Re-implementação independente** (`validation/mathematica_reference.py`): MOD0d
   digitado diretamente das equações extraídas — geometria, Double Wiebe (forma
   `If[θ≥θ0, …]` sem corte em θ0+Δθ), Hohenberg, sistema de 3 EDOs, ICs
   `P(θi)=P1, Tg=T1, Qp=0`, integração `solve_ivp` DOP853 (rtol=atol=1e-10) e
   `erro = sqrt(SSres/(q−2))`. **Nenhuma linha do pacote `double_wiebe` foi reutilizada.**
3. **Comparação dupla**: cada vetor de referência foi avaliado (a) pela re-implementação
   independente e (b) pelo pipeline completo do pacote (`run_simulation` →
   `EngineConfig`/`WiebeParameters`/`SimulationConfig`). A comparação ponto a ponto
   (459 pontos) usa as duas soluções das EDOs.
4. **Âncoras externas**: os dois erros globais produzidos pelo Mathematica
   (100.538 kPa e 55.381 kPa) servem de validação independente das equações extraídas.

**Limitação declarada**: não há Wolfram Engine no ambiente (sem `wolframscript` /
`WolframKernel`). A coluna "Mathematica" da comparação ponto a ponto é, portanto, a
re-implementação independente das equações extraídas — não a execução do notebook em si.
Essa limitação é compensada pelas âncoras numéricas do próprio notebook (ambas
reproduzidas com |Δ| < 3e-4 kPa) e declarada em todas as seções relevantes.

---

## 3. Dados Experimentais (etapa 2)

| Grandeza | Esperado (notebook) | Obtido | Situação |
|---|---:|---:|---|
| Linhas no arquivo | 503 | 503 | ✔ |
| Pontos após filtro −2≤θ≤2 | 459 | **459** | ✔ |
| θ mínimo [rad] | −1.998401994 | −1.998401994 | ✔ |
| θ máximo [rad] | 1.998401994 | 1.998401994 | ✔ |
| Passo médio [rad] | 0.008726646 | 0.008726646 | ✔ |
| P1 [kPa] (bar×100) | 138.2 | 138.2 | ✔ |

A série produzida por `read_table`+`prepare_series` (radianos, bar, filtro −2..2,
ordenação) é **bit a bit idêntica** à extração do notebook. Conforme a especificação,
nenhuma correção ou reamostragem foi aplicada.

## 4. Constantes e Geometria (etapas 3-4)

- **14 grandezas** comparadas entre as constantes do MOD0d e `EngineConfig`
  (d, s, l, ω, κ, Acil, Vp, r, R, Vd, mcomb, PCI, T1, Tw): todas idênticas
  (rtol 1e-10). Ver `validation_metrics.json` → etapa 3.
- **Geometria em 8 ângulos** (V, dV/dθ, y, As): max|Δ| = 0 (atol 1e-12) contra a
  re-implementação; dV/dθ validada também por diferença central (erro relativo 3.6e-10).
- Verificação física: V(θ=0) = Vd/(Rc−1) (volume de folga no TDC) — exato.

## 5. Double Wiebe (etapa 5)

- **Zeros antes do início**: x1, x2, dx1, dx2 = 0 exatamente para θ < θ0 (máscara do
  pacote equivale ao `If[θ≥θ0, …]` do notebook) — max|·| = 0.
- **Potências fracionárias negativas**: com m=0.3 (z negativo fora da máscara), nenhuma
  saída NaN/inf — o padrão `active` + `z_safe` do pacote reproduz a semântica do `If`
  do Mathematica (não é `np.where` puro, que avaliaria os dois ramos).
- **Derivadas** dx1/dθ e dx2/dθ: diferença central com erro relativo ≤ 3.2e-10 no
  interior das fases.
- **Combinação**: x = β·x1 + (1−β)·x2 — o `alpha` do Python multiplica a fase 1, como o
  `β` do notebook (ordem MOD0d: Rc, m1, θ01, Δθ1, m2, θ02, Δθ2, β).
- **Redução ao single**: β=1 → x = x1; β=0 → x = x2 (exato, 1e-14).
- **Modo contínuo** (cauda assintótica após θ0+Δθ, sem corte) é o default — igual ao
  notebook. O modo `bounded` do Python é extensão não presente no notebook (inativo
  por padrão, sem efeito nestes testes).

## 6. Calor Liberado e Hohenberg (etapas 6-7)

- Q_total = mcomb·PCI = **0.369477802 kJ/ciclo** (idêntico); dQ1+dQ2 = dQ_total verificado
  (max |Δ| = 1e-12) e ∫dQ/dθ dθ ≈ Q_total (cauda assintótica integrada, |Δ| < 1e-2 kJ).
- **Hohenberg** h = 130·V^(−0.06)·(P·10⁻²)^0.8·Tg^(−0.4)·(Vp+1.4)^0.8: max|Δh| = 0
  contra a re-implementação em varredura de 200 pontos (P: 138.2–9000 kPa;
  Tg: 308.15–2600 K). O fator P·10⁻² (kPa→bar) está correto nos dois lados.

## 7. EDOs e Casos-Âncora (etapas 8-11)

Sistema integrado com `solve_ivp` DOP853, `t_eval` nos 459 ângulos experimentais
(sem interpolação), ICs idênticos. Convergência verificada apertando tolerâncias
(rtol/atol 1e-9 → 1e-10 → 1e-11): Δerro máximo = **1.0e-4 kPa** — as tolerâncias são
adequadas.

| Caso | Vetor (Rc, m1, θ01, Δθ1, m2, θ02, Δθ2, β) | âncora [kPa] | referência [kPa] | Python [kPa] | Δref | Δpy | max\|ΔP\| ref-py |
|---|---|---:|---:|---:|---:|---:|---:|
| **A** obrigatório | 15.34; 0.6; −8°; 15°; 0.141; 2.83°; 62.06°; 0.063 | 100.538 | 100.537817 | **100.537707** | −0.000183 | −0.000293 | 7.5e-4 kPa |
| **B** PSO k=190 | 15.3496; 0.6; −0.193038; 0.436332; 0.9; 0.0317214; 0.523599; 0.075285 | 55.381 | 55.380920 | **55.380933** | −0.000080 | −0.000067 | 2.4e-4 kPa |
| **C** comentado k=532 (auxiliar) | 15.340724668321869; 0.5999990714343; −0.13962634015954636; 0.2617993877991494; 0.14162900274166848; 0.04937752133906362; 1.082970478069876; 0.06333317846901508 | 34.25858860927897 | 100.557508 | **100.557384** | +66.299 | +66.299 | 9.0e-4 kPa |

- **Casos A e B**: aprovados com margem ~1700×–7500× dentro da tolerância de 0.5 kPa.
  Os erros relativo e absoluto ponto a ponto entre a referência independente e o
  pipeline Python ficam 3 ordens de grandeza abaixo dos limites (1.0 kPa / 1.0 K / 0.1%).
  A trajetória estocástica do PSO não precisou ser reproduzida (conforme a
  especificação): os vetores foram avaliados diretamente.
- **Caso C (auxiliar)**: o valor 34.25858860927897 provém de uma **célula comentada** do
  notebook (PSO com k=532). Com as equações do MOD0d **ativo**, o mesmo vetor produz
  100.5575 kPa — e a re-implementação independente dá exatamente o mesmo valor
  (Δ ref-vs-python = 1.2e-4 kPa), ou seja, o Python não divergiu do MOD0d ativo; é o
  resultado comentado que é inconsistente com ele. **Investigação exaustiva** (todas as
  variantes a seguir produziram valores entre 93 e 826 kPa, nunca 34.2586):
  calor desligado; β no outro ramo; kp=1.35/1.4; a1=a2 ∈ {3, 4.6, 5}; Tw=300; T1=300;
  dados completos sem filtro (503 pontos, erro 93.21); permutações da ordem dos
  parâmetros. Hipóteses restantes: estado anterior das equações do MOD0d quando aquela
  célula executou, ou arquivo experimental diferente (a saída é de uma execução
  histórica — precisamente o tipo de conteúdo comentado que a especificação manda
  tratar apenas como fonte técnica e identificar como teste auxiliar). **Não é critério
  obrigatório**; os critérios quantitativos obrigatórios (A e B) foram aprovados.

## 8. Função de Erro (etapa 9)

- **Notebook**: `erro = sqrt(SSres/(q−2))`, q = 459 — **preservado** (o divisor q−2.0
  não foi alterado em nenhum momento do teste; a re-implementação usa exatamente essa
  expressão e o caso A ancora em 100.537817 kPa).
- **Pacote Python**: `metrics.rmse = sqrt(SSE/q)` — **divergência documentada**
  (divisor q vs q−2). Razão esperada sqrt(q/(q−2)) = 1.002185795, medida 1.002185795
  (caso A: erro notebook 100.537817 vs RMSE pacote 100.318541). Ambos os valores são
  reportados em **colunas separadas** em `validation_metrics.json`
  (`erro_notebook_casoA` e `rmse_python_casoA`), e o teste
  `test_funcao_erro_divisor_q_menos_2` fixa a relação exata entre as duas.
- Recomendação (não aplicada — nenhum código foi modificado): quando for preciso
  comparar diretamente com saídas do notebook, usar
  `sqrt(SSres/(q−2))` (disponível em `validation/mathematica_reference.erro_notebook`);
  eventualmente expor essa métrica no pacote com nome próprio
  (ex.: `rmse_dof`), mantendo `rmse` como está.

## 9. Comparação Ponto a Ponto (etapa 12)

`pointwise_comparison.csv` — 459 linhas (caso A, obrigatório), colunas:
`theta_rad, theta_deg, P_mathematica_kPa, P_python_kPa, erro_P_kPa,
erro_P_percentual, Tg_K, Qp_J, x1, x2, x_total, dx1, dx2, dx_total,
V_m3, dV_dtheta_m3_rad, As_m2`.

| Indicador (caso A) | Valor | Limite | Situação |
|---|---:|---:|---|
| max \|P_math − P_python\| | 7.53e-4 kPa | 1.0 kPa | ✔ |
| max erro percentual | 2.4e-3 % | 0.1% | ✔ |
| max \|Tg_math − Tg_python\| | 1.41e-4 K | 1.0 K | ✔ |

O erro percentual máximo relativo ao Mathematica é dominado pela região de baixa
pressão (θ < −1 rad), onde |ΔP| < 0.2 kPa em valores absolutos.

## 10. Gráficos (etapa 13)

- `validation_pressure.png` — experimental (`+`, vermelho), Mathematica (linha preta
  tracejada), Python (linha verde), θ em rad, P em kPa. As curvas Mathematica e Python
  se sobrepõem (a verde cobre a preta — coerente com max|ΔP| = 7.5e-4 kPa).
- `validation_residuals.png` — resíduos P_exp−P_math (preto tracejado),
  P_exp−P_python (verde) e a diferença Python−Mathematica (azul), que permanece
  visualmente zero em todo o intervalo.
- `validation_wiebe.png` — x1, x2, x total das duas implementações sobrepostas, com
  inícios de fase marcados pelos degraus de partida.

Conforme exigido, a equivalência **não** foi concluída pelas semelhanças visuais: cada
gráfico é a renderização das métricas numéricas das seções 7 e 9.

## 11. Diagnóstico Hierárquico das Divergências (etapa 14)

**D1 — Função de erro (divisor q vs q−2)** — *divergente, documentada*.
`metrics.rmse = sqrt(SSE/q)` vs notebook `sqrt(SSres/(q−2))`. Causa: generalização do
pacote para métricas convencionais de RMSE. Impacto: fator fixo sqrt(459/457) =
1.0021858 (+0.22%) no valor do erro. **Correção recomendada** (não aplicada): expor a
métrica do notebook como função própria e usá-la quando o alvo for reproduzir saídas
do Mathematica. Nenhum critério de paridade foi avaliado com a métrica errada.

**D2 — Limites de calibração** — *divergente, generalização intencional*.
Notebook: Rc[15,16], m1[0.05,0.6], θ01[−16°,0°], Δθ1[5°,25°], m2[0.9,2.5], θ02[0°,8°],
Δθ2[30°,70°], β[0.05,0.6]. Python (PARAM_SPECS): Rc[14,20], θ01[−30°,10°],
Δθ1[2°,40°], m1[0.05,3], a1[1,10], θ02[−20°,40°], Δθ2[5°,90°], m2[0.05,3], a2[1,10],
α[0.05,0.95] — superset que inclui os limites do notebook (com m1/Δθ1 mais largos).
Impacto: nenhum sobre a avaliação de vetores fixos; apenas uma busca PSO iniciada de
outro ponto pode convergir a um mínimo diferente. **Correção recomendada**: acrescentar
um perfil "notebook" de limites (presets) para reprodução 1:1 do notebook, mantendo os
atuais como default generalizado.

**D3 — Ordem dos parâmetros** — *divergente por design, mapeamento verificado*.
MOD0d: (Rc, m1, θ01, Δθ1, m2, θ02, Δθ2, β); PARAM_ORDER: (Rc, θ01, δ1, m1, a1, θ02,
δ2, m2, a2, α). As equações são as mesmas; β ≡ α multiplicando a fase 1. Sem correção
necessária.

**D4 — Resultado comentado k=532 (34.2586 kPa)** — *não reproduzível pelo MOD0d ativo*.
Causa: saída histórica de célula comentada, incompatível com as equações ativas do
próprio notebook (a re-implementação independente e o Python concordam entre si em
1.2e-4 kPa e ambas dão ~100.5575 para o vetor). Variantes testadas não reproduzem o
valor (§7). **Correção recomendada**: nenhuma no Python; tratar o valor apenas como
registro histórico (teste auxiliar identificado, sem valor de paridade).

**D5 — Ausência de Wolfram Engine** — *não verificável no ambiente*.
Causa: ambiente sem wolframscript/WolframKernel. Mitigação aplicada: re-implementação
independente + âncoras numéricas do próprio notebook (A e B), ambas reproduzidas com
|Δ| < 3e-4 kPa. Correção recomendada: executar este mesmo harness com o notebook em um
ambiente com Wolfram Engine para eliminar a ressalva (o harness já produz os vetores e
espera os valores das colunas "Mathematica").

## 12. Conclusão Objetiva

**VALIDADO COM RESSALVAS.**

A implementação Python reproduz o MOD0d do notebook dentro de todas as tolerâncias
especificadas:
- caso obrigatório: 100.537707 kPa vs 100.538 kPa (|Δ| = 0.000293 kPa ≤ 0.5);
- vetor ativo do PSO: 55.380933 kPa vs 55.381 kPa (|Δ| = 0.000067 kPa ≤ 0.5);
- ponto a ponto: max|ΔP| = 7.5e-4 kPa ≤ 1.0; max|ΔTg| = 1.4e-4 K ≤ 1.0;
  erro relativo máximo 2.4e-3% ≤ 0.1%;
- dados, constantes, geometria, Double Wiebe, calor, Hohenberg, ICs e a função de erro
  do notebook (com divisor q−2 **preservado**): idênticos ou numericamente equivalentes.

Ressalvas que impedem o "VALIDADO" pleno: (i) sem Wolfram Engine, a paridade ponto a
ponto se apoia em re-implementação independente ancorada nos resultados do notebook —
não na execução direta do `.nb`; (ii) a âncora auxiliar do vetor comentado (34.2586 kPa,
célula comentada k=532) não é reproduzível pelo MOD0d ativo (inconsistência interna do
notebook, documentada com a investigação completa); (iii) `metrics.rmse` do pacote usa
divisor q (comparado em coluna separada, sem afetar os critérios); (iv) limites de
calibração generalizados (superset intencional). Nenhum arquivo do pacote foi
modificado durante o diagnóstico.

---

### Artefatos produzidos

| Arquivo | Conteúdo |
|---|---|
| `validation/equation_traceability.csv` | Matriz de rastreabilidade (50 componentes: expressão Mathematica, expressão Python, unidades, status, observação — 36 idênticos, 6 numericamente equivalentes, 5 divergentes, 2 não implementados no notebook, 1 não verificável) |
| `validation/pointwise_comparison.csv` | 459 pontos: P, Tg, Qp, x1/x2/x, dx1/dx2/dx, V, dV/dθ, As das duas implementações |
| `validation/validation_metrics.json` | 63 checagens numeradas por etapa + casos-âncora + função de erro + limitação declarada |
| `validation/validation_pressure.png` | Pressão: experimental/Mathematica/Python |
| `validation/validation_residuals.png` | Resíduos e diferença entre implementações |
| `validation/validation_wiebe.png` | Frações queimadas das duas implementações |
| `validation/mathematica_reference.py` | Re-implementação independente do MOD0d |
| `validation/run_validation.py` | Harness executável (etapas 2-13) |
| `tests/test_mathematica_parity.py` | 15 testes de paridade (15 passed; suíte completa: 91+15) |