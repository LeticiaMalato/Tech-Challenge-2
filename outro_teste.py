import pandas as pd

train = pd.read_parquet("data/features/train.parquet")
test  = pd.read_parquet("data/features/test.parquet")

test_users  = set(test["visitorid"].unique())
train_users = set(train["visitorid"].unique())

ausentes = test_users - train_users
presentes_sem_filtro = test_users & train_users


counts = train.groupby("visitorid").size()
removidos_pelo_filtro = {
    u for u in presentes_sem_filtro
    if counts.get(u, 0) < 3
}

print(f"Usuários no teste:                    {len(test_users)}")
print(f"Ausentes do treino completo:          {len(ausentes)}")
print(f"Presentes mas removidos pelo filtro:  {len(removidos_pelo_filtro)}")
print(f"Interações dos removidos pelo filtro:")
print(counts[counts.index.isin(removidos_pelo_filtro)].describe())