import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 100

data = {
    'Вік': np.random.randint(21, 65, n_samples),
    'Дохід_тис': np.random.randint(15, 150, n_samples),
    'Кредитна_історія': np.random.randint(0, 2, n_samples),
    'Стаж_роботи_роки': np.random.randint(0, 40, n_samples),
    'Сума_кредиту_тис': np.random.randint(10, 500, n_samples)
}
df = pd.DataFrame(data)

conditions = (df['Кредитна_історія'] == 1) & (df['Дохід_тис'] > 30) & (df['Сума_кредиту_тис'] < df['Дохід_тис'] * 12)
df['Видача_кредиту'] = np.where(conditions, 1, 0)
flip_idx = np.random.choice(df.index, size=10, replace=False)
df.loc[flip_idx, 'Видача_кредиту'] = 1 - df.loc[flip_idx, 'Видача_кредиту']

X = df.drop('Видача_кредиту', axis=1)
y = df['Видача_кредиту']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Результат навчання")
print(f"Точність моделі на тестових даних: {acc:.2f} (або {acc*100}%)\n")

print("Важливість ознак")
importances = clf.feature_importances_
for feature, imp in zip(X.columns, importances):
    print(f"Ознака '{feature}': {imp:.3f}")

plt.figure(figsize=(22, 10))
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=['Відмова', 'Видача'], 
          filled=True, 
          rounded=True,
          fontsize=12,
          proportion=False)
plt.title("Система схвалення кредитів", fontsize=16)
plt.tight_layout()
plt.show()