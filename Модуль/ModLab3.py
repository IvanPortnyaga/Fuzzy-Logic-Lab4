import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
time_steps = np.arange(0, 250)
time_series = np.sin(0.1 * time_steps) + 0.05 * time_steps + np.random.normal(0, 0.2, 250)

window_size = 5
X = []
y = []
for i in range(len(time_series) - window_size):
    X.append(time_series[i:i + window_size])
    y.append(time_series[i + window_size])

X = np.array(X)
y = np.array(y)

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

t_test = time_steps[window_size + split_index:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
lin_pred = lin_reg.predict(X_test_scaled)

nn_reg = MLPRegressor(hidden_layer_sizes=(20, 10), solver='lbfgs', alpha=1.0, max_iter=5000, random_state=42)
nn_reg.fit(X_train_scaled, y_train)
nn_pred = nn_reg.predict(X_test_scaled)

print("Оцінка моделі Множинної Лінійної Регресії")
print(f"Середня абсолютна похибка (MAE): {mean_absolute_error(y_test, lin_pred):.4f}")
print(f"Коефіцієнт детермінації (R^2):   {r2_score(y_test, lin_pred):.4f}\n")

print("Оцінка моделі Штучної Нейронної Мережі")
print(f"Середня абсолютна похибка (MAE): {mean_absolute_error(y_test, nn_pred):.4f}")
print(f"Коефіцієнт детермінації (R^2):   {r2_score(y_test, nn_pred):.4f}")

plt.figure(figsize=(14, 7))
t_train = time_steps[window_size:window_size + split_index]
plt.plot(t_train, y_train, color='gray', alpha=0.5, label='Історичні дані (Навчальна вибірка)')
plt.plot(t_test, y_test, color='black', linewidth=2, label='Реальні дані (Тестова вибірка)')
plt.plot(t_test, lin_pred, color='blue', linestyle='--', linewidth=2, label='Прогноз: Лінійна Регресія')
plt.plot(t_test, nn_pred, color='red', linestyle='-.', linewidth=2, label='Прогноз: Нейронна Мережа')
plt.title('Прогнозування часового ряду: Лінійна регресія vs Нейронна мережа', fontsize=16)
plt.xlabel('Час', fontsize=12)
plt.ylabel('Значення показника', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.show()