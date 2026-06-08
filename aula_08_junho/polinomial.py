import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Carregar dados
df = pd.read_csv("dataset.csv")

X = df[["age", "experience"]]
y = df["income"]

# Criar variáveis polinomiais de grau 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_poly, y)

# Resultados
print("Intercepto:", modelo.intercept_, "\n")  # ponto de partida da curva de regressão.
print("Coeficientes:", modelo.coef_, "\n")     # impacto de cada variável independente na variável dependente.
print("R²:", modelo.score(X_poly, y))          # quanto + perto de 1, melhor.