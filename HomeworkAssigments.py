import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
X, y = make_regression(n_samples=2000, n_features=4, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_none = r2_score(y_test, y_pred)
print("Завдання 1:")
table1 = pd.DataFrame({
    "Ознака": [f"x{i+1}" for i in range(4)],
    "Коефіцієнт": model.coef_
})
print(table1.to_string(index=False))
print(f"R² = {r2_none}\n")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)
print("Завдання 2:")
table2 = pd.DataFrame({
    "Масштабування": ["Ні", "StandardScaler"],
    "R²": [r2_none, r2_scaled]
})
print(table2.to_string(index=False))
print("\n" + "="*40 + "\n")
np.random.seed(42)
mask = np.random.rand(X_train.shape[0]) < 0.1
X_train_missing = X_train.copy()
X_train_missing[mask, 0] = np.nan
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_missing)
X_test_imputed = imputer.transform(X_test)
model_imputed = LinearRegression()
model_imputed.fit(X_train_imputed, y_train)
r2_imputed = r2_score(y_test, model_imputed.predict(X_test_imputed))
print("Завдання 3:")
table3 = pd.DataFrame({
    "Сценарій": ["Без пропусків", "Після ім'ютації"],
    "R²": [r2_none, r2_imputed]
})
print(table3.to_string(index=False))
print("\n" + "="*40 + "\n")
X4, y4 = make_regression(n_samples=2000, n_features=4, noise=15, random_state=42)
X4_scaled = scaler.fit_transform(X4)
model4 = LinearRegression()
scores = cross_val_score(model4, X4_scaled, y4, cv=5, scoring='r2')
print("Завдання 4:")
print(f"Середнє значення R²: {np.mean(scores):.4f}")
print(f"Стандартне відхилення R²: {np.std(scores):.4f}")