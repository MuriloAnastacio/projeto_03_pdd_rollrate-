import pandas as pd
import numpy as np

np.random.seed(42)

# -------------------------
# Simulação de carteira
# -------------------------

n = 5000

carteira = pd.DataFrame({
    "cliente_id": range(1, n + 1),
    "saldo_devedor": np.random.gamma(2.5, 2000, n),
    "dias_atraso": np.random.choice(
        [0, 15, 45, 75, 120],
        size=n,
        p=[0.60, 0.15, 0.10, 0.08, 0.07]
    ),
    "taxa_juros": np.random.uniform(0.02, 0.08, n)
})

# Bucketização
def bucket_dpd(dpd):
    if dpd == 0:
        return "A - Adimplente"
    elif dpd <= 30:
        return "B - 1-30"
    elif dpd <= 60:
        return "C - 31-60"
    elif dpd <= 90:
        return "D - 61-90"
    else:
        return "E - 90+"

carteira["bucket"] = carteira["dias_atraso"].apply(bucket_dpd)

# -------------------------
# Matriz de Roll Rate
# -------------------------

transicoes = {
    "A - Adimplente": [0.85, 0.10, 0.03, 0.01, 0.01],
    "B - 1-30": [0.30, 0.40, 0.20, 0.07, 0.03],
    "C - 31-60": [0.10, 0.20, 0.40, 0.20, 0.10],
    "D - 61-90": [0.05, 0.10, 0.20, 0.40, 0.25],
    "E - 90+": [0.02, 0.03, 0.05, 0.10, 0.80]
}

estagios = list(transicoes.keys())

matriz_rollrate = pd.DataFrame(transicoes, index=estagios)

# -------------------------
# Estimativa de PD
# -------------------------

pd_bucket = matriz_rollrate.loc["E - 90+"]
pd_estimada = pd_bucket / pd_bucket.sum()

# -------------------------
# Cálculo PDD
# -------------------------

LGD = 0.75  # Loss Given Default

carteira["pd_aplicada"] = carteira["bucket"].map(pd_estimada)
carteira["lgd"] = LGD
carteira["ead"] = carteira["saldo_devedor"]

carteira["pdd_estimado"] = (
    carteira["pd_aplicada"] *
    carteira["lgd"] *
    carteira["ead"]
)

# -------------------------
# Resultado Consolidado
# -------------------------

resumo = carteira.groupby("bucket").agg(
    saldo_total=("saldo_devedor", "sum"),
    pdd_total=("pdd_estimado", "sum"),
    clientes=("cliente_id", "count")
)

resumo["coverage_ratio"] = resumo["pdd_total"] / resumo["saldo_total"]

print("\n===== MATRIZ ROLL RATE =====")
print(matriz_rollrate)

print("\n===== RESUMO PDD =====")
print(resumo)

carteira.to_csv("carteira_simulada.csv", index=False)
