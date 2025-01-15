kod da şu
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Veri setlerini yükleme
data = pd.read_csv('data.csv')
ages = pd.read_csv('Ages.csv')

# Feature'ları ve target'ı hazırlama
X = data.iloc[:, 2:].values
y = ages['Age'].values

# Veriyi train ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature'ları ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost modeli
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,  # L1 regularization
    reg_lambda=1, # L2 regularization
    random_state=42
)


# Model eğitimi
xgb_model.fit(X_train_scaled, y_train)

# Tahminler
y_pred_train = xgb_model.predict(X_train_scaled)
y_pred_test = xgb_model.predict(X_test_scaled)

# Performans metrikleri
print("\nEgitim seti sonuclari:")
print(f'Mean Absolute Error: {mean_absolute_error(y_train, y_pred_train):.2f} yil')
print(f'R-squared Score: {r2_score(y_train, y_pred_train):.4f}')

print("\nTest seti sonuclari:")
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred_test):.2f} yil')
print(f'R-squared Score: {r2_score(y_test, y_pred_test):.4f}')

# Tahmin vs Gerçek değerleri görselleştirme (Test seti)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Yaş')
plt.ylabel('Tahmin Edilen Yaş')
plt.title('Gerçek vs Tahmin Edilen Yaş (Test Seti)')
plt.tight_layout()
plt.show()

# En önemli özellikleri görselleştirme (ilk 20)
feature_importance = pd.DataFrame({
    'feature': data.columns[2:],
    'importance': xgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 6))
plt.bar(range(20), feature_importance['importance'])
plt.xticks(range(20), feature_importance['feature'], rotation=45, ha='right')
plt.xlabel('Mikroorganizma')
plt.ylabel('Önem Derecesi')
plt.title('En Önemli 20 Mikroorganizma')
plt.tight_layout()
plt.show()