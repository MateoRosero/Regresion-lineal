import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("SalaryData.csv", sep=";")
df.columns = df.columns.str.strip()

print("Columnas en el archivo:")
print(df.columns.tolist())


X = df[['Years of Experience']]  
y = df['Salary']                 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


modelo = LinearRegression()
modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nResultados del modelo:")
print(f"Pendiente (coeficiente): {modelo.coef_[0]:.2f}")
print(f"Intercepto: {modelo.intercept_:.2f}")
print(f"R² (coeficiente de determinación): {r2:.2f}")


# === Gráfico 1: Datos de entrenamiento ===
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='green', label='Datos de entrenamiento')
plt.plot(X_train, modelo.predict(X_train), color='red', linewidth=2, label='Línea de regresión')
plt.title("Regresión Lineal - Conjunto de Entrenamiento")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.legend()
plt.grid(True)
plt.show()

# === Gráfico 2: Datos de prueba ===
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Datos de prueba')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Línea de regresión')
plt.title("Regresión Lineal - Conjunto de Prueba")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.legend()
plt.grid(True)
plt.show()

