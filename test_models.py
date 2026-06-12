# import pandas as pd
# from src.models.factory import build_recommender
# from src.models.metrics import evaluate_recommender

# train = pd.read_parquet("data/features/train.parquet")
# test  = pd.read_parquet("data/features/test.parquet")

# for name in ["popularity", "item_knn", "bpr_mf", "als"]:
#     print(f"\ntreinando {name}...")
#     model = build_recommender(name)
#     model.fit(train)
#     results = evaluate_recommender(model, test, k_values=[5, 10, 20])
#     print(results.to_string(index=False))

import pandas as pd
from src.models.factory import build_recommender
from src.models.metrics import evaluate_recommender

train = pd.read_parquet("data/features/train.parquet")
test  = pd.read_parquet("data/features/test.parquet")

# amostra completa para modelos rápidos
test_full = test[test["visitorid"].isin(
    test["visitorid"].drop_duplicates().sample(2000, random_state=42)
)]

# amostra menor só para o item_knn
test_knn = test[test["visitorid"].isin(
    test["visitorid"].drop_duplicates().sample(500, random_state=42)
)]

modelos_rapidos = ["popularity", "bpr_mf", "als"]
for name in modelos_rapidos:
    print(f"\ntreinando {name} (2000 usuários)...")
    model = build_recommender(name)
    model.fit(train)
    results = evaluate_recommender(model, test_full, k_values=[5, 10, 20])
    print(results.to_string(index=False))

print(f"\ntreinando item_knn (500 usuários — mais lento)...")
model = build_recommender("item_knn")
model.fit(train)
results = evaluate_recommender(model, test_knn, k_values=[5, 10, 20])
print(results.to_string(index=False))