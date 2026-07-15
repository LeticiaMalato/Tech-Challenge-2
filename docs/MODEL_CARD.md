# Model Card — Sistema de Recomendação RetailRocket

## Detalhes do modelo

- **Nome**: `retailrocket-recommender`
- **Tipo**: Recomendador colaborativo via Matrix Factorization com bias
  (`score(u,i) = <emb_u, emb_i> + b_u + b_i`), treinado discriminativamente
  com negative sampling e `BCEWithLogitsLoss`
- **Framework**: PyTorch
- **Versão em produção**: v4 no MLflow Model Registry (estágio *Production*)
- **Desenvolvido por**: Leticia Malato, Rafael Maranhão — Tech Challenge,
  Fase 2, Pós-Tech (FIAP)
- **Licença dos dados**: dataset público RetailRocket (Kaggle)

O nome interno da classe (`MLPRecommender`) é uma referência histórica ao
desafio original (que sugere "MLP ou embedding-based"); a arquitetura
implementada é Matrix Factorization, não uma rede com camadas ocultas —
ver seção "Decisões de arquitetura" para a justificativa.

## Uso pretendido

**Caso de uso primário**: recomendação de produtos em um catálogo de
e-commerce, a partir do histórico de navegação (view/addtocart/transaction)
de um usuário já conhecido no período de treino.

**Fora de escopo**:
- **Cold-start de usuário ou item novo**: o modelo não tem embedding para
  IDs fora do vocabulário de treino. Usuários/itens novos precisam de
  re-treino ou de uma estratégia de fallback (ex: `PopularityRecommender`
  como default).
- **Recomendação em tempo real com re-treino contínuo**: o pipeline atual
  é batch (treino offline, checkpoint estático); não há mecanismo de
  atualização incremental de embeddings.
- **Domínios fora de e-commerce**: os pesos de evento (`view=1`,
  `addtocart=2`, `transaction=3`) são específicos do padrão de navegação
  de varejo online.

## Dados de treinamento

- **Fonte**: RetailRocket (eventos de navegação, árvore de categorias,
  propriedades de itens)
- **Volume bruto**: ~2,75 milhões de eventos
- **Pós k-core (k=5) + split temporal + filtro de cobertura**: ~625 mil
  interações de treino, 54.612 usuários, 34.723 itens
- **Split**: temporal (corte pelo quantil 0.8 dos timestamps), não
  aleatório — simula o cenário real de prever comportamento futuro a
  partir de dados passados
- **Cold-start no split**: usuários/itens do período de teste que não
  aparecem no treino são removidos da avaliação (34,5% dos usuários do
  período de teste tinham histórico suficiente no treino para serem
  avaliados)
- **Pré-processamento**: deduplicação de eventos exatos, conversão de
  timestamp para UTC, merge *as-of* com propriedades de item (evita
  vazamento de informação futura — a propriedade usada é sempre a mais
  recente disponível até o momento do evento)

## Decisões de arquitetura

**Por que Matrix Factorization em vez de MLP com camadas ocultas?**
Datasets de e-commerce são esparsos — poucos exemplos por item —, cenário
em que camadas ocultas tendem a overfittar antes de extrair sinal útil.
MF com bias é equivalente em capacidade ao SVD, mas treinada
discriminativamente com negative sampling, capturando melhor o sinal de
"o que o usuário NÃO interagiu" do que a fatoração puramente linear do SVD.

**Por que o LogisticRecommender usa features manuais em vez de embeddings?**
Testa a hipótese alternativa de que sinais diretos e interpretáveis
(popularidade do item, atividade do usuário, afinidade) seriam suficientes
para personalização — mantendo uma família de baseline
arquiteturalmente independente do SVD/MLP (ambos baseados em fatoração
latente).

## Procedimento de treino

| Hiperparâmetro | Valor |
|---|---|
| `embed_dim` | 32 |
| `neg_ratio` | 4 |
| Learning rate | 5e-4 |
| Weight decay | 1e-2 (AdamW) |
| Batch size | 2048 |
| Max epochs | 50 |
| Patience (early stopping) | 10 |
| Seed | 42 (fixada em todos os geradores: `random`, `numpy`, `torch`, `cudnn`) |

- **Split treino/validação interno**: leave-one-out temporal por usuário
  (o evento mais recente de cada usuário vai para validação)
- **Early stopping**: monitorado por `val_loss`, restaurando os pesos da
  melhor época
- **Execução mais recente**: parou por early stopping na epoch 35,
  restaurando os pesos da epoch 25 (`val_loss=0.3044`) — passado esse
  ponto, `train_loss` continuou caindo mas `val_loss` voltou a subir,
  sinal claro de overfitting a partir da epoch ~26

## Avaliação

**Métricas**: Hit Rate, Precision, Recall, NDCG e MRR, calculadas em
k ∈ {5, 10, 20}. Ground truth restrito a eventos `addtocart` e
`transaction` (não `view`), para focar em intenção comportamental forte
em vez de cliques exploratórios. MRR é destacado como a métrica mais
alinhada ao caso de uso — em e-commerce, o usuário tipicamente escaneia a
lista de cima para baixo e para no primeiro item de interesse.

**Resultados no conjunto de teste** (split temporal, execução via
`dvc repro`):

| Modelo | hit_rate@20 | precision@20 | recall@20 | ndcg@20 | mrr@20 |
|---|---|---|---|---|---|
| **MLP (neural)** | **0,0656** | **0,0044** | **0,0243** | **0,0154** | **0,0228** |
| ItemKNN | 0,0378 | 0,0025 | 0,0124 | 0,0081 | 0,0117 |
| Popularity | 0,0235 | 0,0015 | 0,0073 | 0,0046 | 0,0061 |
| Logistic | 0,0203 | 0,0013 | 0,0065 | 0,0038 | 0,0049 |
| SVD | 0,0126 | 0,0007 | 0,0039 | 0,0024 | 0,0036 |

**Interpretação**:
- O MLP supera o melhor baseline (ItemKNN) em **~74% de hit_rate@20** e
  vence em todas as métricas, em todos os valores de k.
- **Achado mais relevante**: SVD e Logistic ficam *abaixo* da
  Popularity — a estratégia mais ingênua possível (mesma lista para
  todos os usuários) — em todos os k avaliados. Isso indica que tentar
  personalizar com sinais fracos (fatoração linear pura, ou features
  manuais sem interação profunda) pode piorar a recomendação em vez de
  melhorá-la, num cenário de dados esparsos como este. Só a MF treinada
  discriminativamente (MLP) consegue extrair sinal de personalização que
  supera a baseline não-personalizada.
- **Critério de seleção para o Model Registry**: NDCG@10, por equilibrar
  posição do acerto (como o MRR) com sensibilidade a múltiplos acertos
  relevantes na lista (que o MRR ignora, ao considerar só o primeiro).

## Limitações conhecidas e vieses

- **Viés de popularidade residual**: mesmo o MLP, sendo o melhor modelo,
  ainda recomenda itens populares com mais frequência que itens de cauda
  longa — inerente a qualquer modelo treinado por maximização de
  verossimilhança sobre dados de interação, que naturalmente contêm mais
  exemplos positivos de itens populares. Não foi aplicada nenhuma técnica
  de correção de exposição (ex: re-ranking por diversidade).
- **Leakage técnico no k-core**: o filtro de densidade (k=5) é aplicado
  sobre o dataset inteiro (treino + teste) antes do split temporal — ou
  seja, a decisão de "quais usuários/itens são densos o suficiente" já
  enxerga atividade do período de teste. É uma escolha comum em
  pipelines de recsys, mas tecnicamente um vazamento de informação;
  documentado aqui como trade-off consciente, não como bug.
- **Comportamento inconsistente entre baselines para usuário
  desconhecido**: `SVDRecommender` retorna lista vazia; `LogisticRecommender`
  retorna `k` itens com score `-inf` em ordem arbitrária. Nenhum dos dois
  quebra a aplicação, mas a interface `Recommender` não garante contrato
  único de comportamento nesse caso de borda entre implementações.
- **Encoders de feature engineering não reaproveitados pelos modelos**:
  os artefatos `user_encoder.pkl`/`item_encoder.pkl` gerados no stage
  `feature_eng` não são consumidos pelos recomendadores — cada um
  reconstrói seu próprio índice local dentro do `fit()`. Isso não afeta
  as métricas reportadas (tudo roda no mesmo processo), mas significa que
  esses artefatos não são diretamente reutilizáveis para servir um
  checkpoint isoladamente em produção, sem re-treinar.
- **Ausência de cobertura/diversidade nas métricas de avaliação**: as
  métricas usadas medem relevância (o modelo acerta o que o usuário
  quis?), mas não medem quão diverso ou quão bem distribuído pelo
  catálogo é o conjunto de recomendações — um modelo pode performar bem
  nessas métricas recomendando repetidamente um subconjunto pequeno de
  itens populares.
- **API do Model Registry deprecada**: a transição de estágio
  (`transition_model_version_stage`, Staging→Production) está deprecada
  desde o MLflow 2.9 em favor de aliases (`set_registered_model_alias`),
  mas permanece funcional. Usada aqui por corresponder à nomenclatura
  pedida no desafio (Staging → Production); uma versão futura deveria
  migrar para aliases.

## Considerações éticas

- **Dados anonimizados**: o dataset usa `visitorid` como identificador
  anônimo, sem PII (nome, e-mail, endereço). Nenhuma informação
  demográfica ou sensível foi usada como feature.
- **Risco de reforço de desigualdade de exposição**: como qualquer
  recomendador colaborativo, o sistema tende a reforçar a visibilidade de
  itens já populares ("rich get richer"), o que pode ser prejudicial para
  vendedores/produtos novos num marketplace real. Mitigação recomendada
  para produção: re-ranking com penalização de popularidade ou reserva de
  slots para exploração.

## Recomendações para uso e evolução futura

- Antes de promover uma nova versão a Production, validar que o
  `ndcg@10` da nova versão supera a versão atual em produção (hoje a
  seleção é feita apenas entre os candidatos de uma única execução, sem
  comparação formal contra o modelo já em produção).
- Para deploy real, resolver a limitação dos encoders não reaproveitados
  (serializar o mapeamento `visitorid`/`itemid` → índice junto do
  checkpoint do modelo escolhido, não apenas dos dados de feature
  engineering).
- Monitorar drift de distribuição de itens/usuários ao longo do tempo,
  já que o modelo foi treinado sobre uma janela temporal fixa do
  RetailRocket, sem mecanismo de re-treino incremental.