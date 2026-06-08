import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

# ==================================================
# CARREGAMENTO DO DATASET
# ==================================================

df = pd.read_csv("dataset.csv")

# Variáveis independentes (X) e dependente (y)
X = df[["age", "experience"]]
y = df["income"]

# Separação treino e teste (utilizada por todos os modelos)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# 1. REGRESSÃO LINEAR
# ==================================================

print("\n" + "="*50)
print("REGRESSÃO LINEAR")
print("="*50)

# Padronização
scaler_linear = StandardScaler()

X_train_linear = scaler_linear.fit_transform(X_train)
X_test_linear = scaler_linear.transform(X_test)

# Modelo
modelo_linear = LinearRegression()
modelo_linear.fit(X_train_linear, y_train)

# Coeficientes e intercepto
print("\nCoeficientes:")
for nome, coef in zip(X.columns, modelo_linear.coef_):
    print(f"{nome}: {coef:.4f}")

print(f"\nIntercepto: {modelo_linear.intercept_:.4f}")

# Previsões
y_pred_linear = modelo_linear.predict(X_test_linear)

# Avaliação
r2_linear = r2_score(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))

print("\nPrevisões:")
print(y_pred_linear)

print(f"\nR²: {r2_linear:.4f}")
print(f"MAE: {mae_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f}")

# ==================================================
# 2. REGRESSÃO POLINOMIAL
# ==================================================

print("\n" + "="*50)
print("REGRESSÃO POLINOMIAL")
print("="*50)

# Padronização
scaler_poly = StandardScaler()

X_train_scaled = scaler_poly.fit_transform(X_train)
X_test_scaled = scaler_poly.transform(X_test)

# Features polinomiais
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print("\nFeatures geradas:")
print(poly.get_feature_names_out(["age", "experience"]))

# Modelo
modelo_poly = LinearRegression()
modelo_poly.fit(X_train_poly, y_train)

# Previsões
y_pred_poly = modelo_poly.predict(X_test_poly)

# Avaliação
r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))

print(f"\nR²: {r2_poly:.4f}")
print(f"MAE: {mae_poly:.4f}")
print(f"RMSE: {rmse_poly:.4f}")

# ==================================================
# 3. ÁRVORE DE DECISÃO
# ==================================================

print("\n" + "="*50)
print("ÁRVORE DE DECISÃO")
print("="*50)

# Árvore não precisa de normalização
modelo_arvore = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)

# Treinamento
modelo_arvore.fit(X_train, y_train)

# Regras aprendidas
regras = export_text(
    modelo_arvore,
    feature_names=list(X.columns)
)

print("\n--- Regras da Árvore ---\n")
print(regras)

# Importância das variáveis
print("\n--- Importância das Features ---\n")
for feat, imp in zip(
    X.columns,
    modelo_arvore.feature_importances_
):
    print(f"{feat}: {imp:.3f}")

# Previsões
y_pred_arvore = modelo_arvore.predict(X_test)

# Avaliação
r2_arvore = r2_score(y_test, y_pred_arvore)
mae_arvore = mean_absolute_error(y_test, y_pred_arvore)
rmse_arvore = np.sqrt(mean_squared_error(y_test, y_pred_arvore))

print("\n--- Resultados ---\n")
print(f"R²: {r2_arvore:.4f}")
print(f"MAE: {mae_arvore:.4f}")
print(f"RMSE: {rmse_arvore:.4f}")

# ==================================================
# RESUMO FINAL
# ==================================================

print("\n" + "="*60)
print("COMPARAÇÃO DOS MODELOS")
print("="*60)

print(
    f"Regressão Linear     -> "
    f"R² = {r2_linear:.4f} | "
    f"MAE = {mae_linear:.4f} | "
    f"RMSE = {rmse_linear:.4f}"
)

print(
    f"Regressão Polinomial -> "
    f"R² = {r2_poly:.4f} | "
    f"MAE = {mae_poly:.4f} | "
    f"RMSE = {rmse_poly:.4f}"
)

print(
    f"Árvore de Decisão    -> "
    f"R² = {r2_arvore:.4f} | "
    f"MAE = {mae_arvore:.4f} | "
    f"RMSE = {rmse_arvore:.4f}"
)