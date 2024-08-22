from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Veri setini yükleme
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
raw_boston = pd.read_csv('C:/Users/MFY/Desktop/A/housing.csv', header=None, delimiter=r"\s+", names=column_names)
X = raw_boston.iloc[:, :-1].values
y = raw_boston.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ridge regresyon modeli
ridge_reg = Ridge()

# GridSearchCV ile alpha değerini seçme
param_grid = {'alpha': [0.049,0.05,0.1,1,5,10,42,50,100]}
grid_search = GridSearchCV(estimator=ridge_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10)
grid_search.fit(X_train, y_train)

# En iyi alpha değeri
best_alpha = grid_search.best_params_['alpha']
print("------------------------------------------")
print("En iyi alpha değeri:", best_alpha)
print("------------------------------------------")

# En iyi alpha değeri ile modeli tekrar eğitme
best_ridge_reg = Ridge(alpha=best_alpha)
best_ridge_reg.fit(X_train, y_train)

# Modelin performansını değerlendirme
y_pred_best = best_ridge_reg.predict(X_test)
best_ridge_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred_best))
best_ridge_R2 = best_ridge_reg.score(X_test, y_test)
best_ridge_MAE = mean_absolute_error(y_test, y_pred_best)
best_ridge_MSE = mean_squared_error(y_test, y_pred_best)

print('MSE metriği:', round(best_ridge_MSE, 4))
print("------------------------------------------")
print('RMSE metriği:', round(best_ridge_RMSE, 4))
print("------------------------------------------")
print('R2 metriği:', round(best_ridge_R2, 4))
print("------------------------------------------")
print('MAE metriği:', round(best_ridge_MAE, 4))
print("------------------------------------------")

