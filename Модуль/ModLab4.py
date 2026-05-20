import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Ініціалізація вхідних змінних (від 0 до 10 балів) та виходу (від 0% до 25% чайових)
food = ctrl.Antecedent(np.arange(0, 11, 1), 'Їжа')
atmosphere = ctrl.Antecedent(np.arange(0, 11, 1), 'Атмосфера')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'Обслуговування')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'Чайові')

# 2. Налаштування функцій належання (нечітких множин)
food.automf(3, names=['погана', 'нормальна', 'смачна'])
atmosphere.automf(3, names=['незатишна', 'звичайна', 'чудова'])
service.automf(3, names=['погане', 'прийнятне', 'відмінне'])

# Для чайових задаємо власні трикутні функції
tip['низькі'] = fuzz.trimf(tip.universe, [0, 0, 10])
tip['середні'] = fuzz.trimf(tip.universe, [5, 12, 18])
tip['високі'] = fuzz.trimf(tip.universe, [15, 25, 25])

# 3. Формування бази експертних правил
rule1 = ctrl.Rule(food['погана'] | service['погане'], tip['низькі'])
rule2 = ctrl.Rule(atmosphere['незатишна'] & service['прийнятне'], tip['низькі'])
rule3 = ctrl.Rule(service['прийнятне'] & food['нормальна'], tip['середні'])
rule4 = ctrl.Rule(service['прийнятне'] & atmosphere['звичайна'], tip['середні'])
rule5 = ctrl.Rule(service['відмінне'] | food['смачна'] | atmosphere['чудова'], tip['високі'])

# 4. Створення та ініціалізація контролера
tip_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
tipping = ctrl.ControlSystemSimulation(tip_ctrl)

# Тестовий розрахунок для конкретного клієнта
tipping.input['Їжа'] = 8.0
tipping.input['Атмосфера'] = 6.5
tipping.input['Обслуговування'] = 9.0
tipping.compute()

print("Результат розрахунку нечіткої системи")
print("Оцінка їжі: 8.0 / 10")
print("Оцінка атмосфери: 6.5 / 10")
print("Оцінка обслуговування: 9.0 / 10")
print(f"Рекомендований розмір чайових: {tipping.output['Чайові']:.2f}%\n")

# 5. Побудова 3D-поверхні для аналізу впливу змінних (Їжа та Обслуговування)
# Фіксуємо атмосферу на середньому рівні (5.0) для візуалізації
x, y = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
z = np.zeros_like(x)

for i in range(20):
    for j in range(20):
        tipping.input['Їжа'] = x[i, j]
        tipping.input['Обслуговування'] = y[i, j]
        tipping.input['Атмосфера'] = 5.0
        tipping.compute()
        z[i, j] = tipping.output['Чайові']

# Візуалізація
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap='plasma', edgecolor='none', alpha=0.9)
ax.set_title('Вплив якості їжі та обслуговування на розмір чайових\n(при середній оцінці атмосфери)', fontsize=15)
ax.set_xlabel('\nЯкість їжі (0-10)', fontsize=12)
ax.set_ylabel('\nОбслуговування (0-10)', fontsize=12)
ax.set_zlabel('\nЧайові (%)', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
plt.tight_layout()
plt.show()