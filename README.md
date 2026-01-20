# Relatório — Simulacao_Otimizacao_IA

## Sumário

Este repositório contém um estudo integrado de **predição de falhas**, **otimização de agendamento** e **simulação estocástica** aplicado ao planejamento de testes veiculares (case fictício). O objetivo principal é reduzir o *makespan* (tempo total de conclusão dos testes) em múltiplas baias, incorporando a probabilidade de falhas prevista por modelos de Machine Learning para gerar agendas mais robustas.

---

## Conteúdo principal

- `Simulacao_Otimizacao_IA.ipynb` — Notebook com todo o fluxo: geração de dados, modelagem preditiva, formulação MILP, execução do solver e simulação Monte Carlo.
- `report.pdf` — Relatório técnico com sumário executivo, metodologia completa, resultados e conclusões (opcional: incluir o PDF original do case).

---

## Estrutura do relatório 

### 1. Introdução e Contexto
Apresenta a motivação do estudo: atrasos em ciclos de testes veiculares ocasionados por falhas inesperadas, necessidade de alocação eficiente de recursos e robustez operacional.

### 2. Objetivos
- Minimizar o Makespan para um conjunto de testes em múltiplas baias.
- Integrar estimativas de risco de falha (probabilidades) ao processo de agendamento.
- Avaliar robustez do cronograma via simulação de Monte Carlo.

### 3. Geração e Análise de Dados
Descreve o dataset (número de testes, features geradas sinteticamente — duração, prioridade, complexidade, temperatura — e variável alvo `failure`). Inclui distribuição estatística resumida, correlações e plots exploratórios gerados pelo notebook.

### 4. Modelo Preditivo de Falhas
- **Algoritmo**: Random Forest Classifier (configuração e validação).
- **Divisão**: treino (70%) / teste (30%).
- **Métricas**: AUC-ROC, Brier Score, matriz de confusão, calibração probabilística.
- **Importância de variáveis**: ranking das features e interpretação.

### 5. Modelo de Otimização (Agendamento Sensível ao Risco)
- **Formulação**: MILP para minimizar Makespan com variáveis de alocação binárias.
- **Estratégia risk-aware**: duração esperada ajustada = duração nominal + p(falha) × tempo_de_reparo.
- **Restrições**: cada teste alocado a exatamente uma baia, capacidade das baias, prioridades, janelas temporais (se aplicável).
- **Solver utilizado**: (ex.: CBC, Gurobi, CPLEX — especificar no notebook).

### 6. Simulação Numérica — Monte Carlo
- **Número de iterações**: (ex.: 2.000)
- **Cenários**: variabilidade natural da duração (ruído gaussiano proporcional) e eventos de falha Bernoulli com p estimado.
- **Métricas de interesse**: tempo médio real, P95, distribuição do makespan real, taxa de retrabalho.

### 7. Resultados e Análises
Resumo dos principais resultados: makespan planejado vs. médio simulado, P95, distribuição de carga entre baias e efeito da estratégia sensível ao risco. Incluir gráficos de Gantt e histogramas (salvos em `results/`).

### 8. Conclusões e Recomendações
Síntese das lições aprendidas, limitações (ex.: dados sintéticos, simplificações do modelo), e próximos passos (ex.: validação com dados reais, incorporação de janelas temporais, otimização multiobjetivo).

---


