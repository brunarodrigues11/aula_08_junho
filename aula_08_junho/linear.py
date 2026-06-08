import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
df = pd.read_csv("dataset.csv")

print("\n--- Regressão Linear ---\n")

# Variáveis independentes e dependente
X = df[["age", "experience"]]
y = df["income"]

# Aplicar o Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_scaled, y)

# Resultados
print("Coeficientes:", modelo.coef_, "\n")     # impacto de cada variável independente na variável dependente.
print("Intercepto:", modelo.intercept_, "\n")  # ponto de partida da linha de regressão.
print("R²:", modelo.score(X_scaled, y), "\n")  # quanto + perto de 1, melhor.


print("\n--- Regressão Polinomial ---\n")


# Variáveis independentes e dependente
X = df[["age", "experience"]]
y = df["income"]

# Aplicar o Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar variáveis polinomiais (grau 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_poly, y)

# Resultados
print("Intercepto:", modelo.intercept_, "\n")  # ponto de partida da curva de regressão.
print("Coeficientes:", modelo.coef_, "\n")     # impacto de cada variável independente na variável dependente.
print("R²:", modelo.score(X_poly, y), "\n")          # quanto + perto de 1, melhor.


print("\n--- Árvore de Decisão ---\n")


X = df[["age", "experience"]]
y = df["income"]

# Criar e treinar a árvore
modelo = DecisionTreeRegressor(random_state=42)
modelo.fit(X, y)

# Fazer previsões
y_pred = modelo.predict(X)

# Calcular R²
r2 = r2_score(y, y_pred)

print("R²:", r2)