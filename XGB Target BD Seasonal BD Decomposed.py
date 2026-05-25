# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:07:40 2026

@author: Asus
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from joblib import dump

# ================= USER SETTINGS =================
station_name = 'Chittagong'  # 🔴 station name (for naming only)

# 🔴 decomposed Tmax file (ONE station per file)
tmax_file = 'D:/Unknown/Research/MATLAB/Daubechies/Temperature/Seasonal/Target BD Stations/Tmax/Monsoon/MRA/Chittagong_merged_data_Monsoon_MODWT.xlsx'

enso_file = 'D:/Unknown/Research/MATLAB/Daubechies/Temperature/Seasonal/Target Nino/All SST(mean)_test_file.xlsx'

season = 'Monsoon'     # 🔴 sheet name
enso_name = 'nino_1+2'     # 🔴 ENSO target
level_name = 'Level3'      # 🔴 CHANGE LEVEL HERE

# ================= OUTPUT SETTINGS =================
output_dir = r'D:/Unknown/Research/MATLAB/Daubechies/Temperature/Seasonal/Target BD Stations/Tmax/Monsoon/XGB/Nino_1+2'   # 🔴 CHANGE THIS PATH

# ================= LOAD DATA =================
df_temp = pd.read_excel(tmax_file)
df_enso = pd.read_excel(enso_file, sheet_name=season)

# ================= INPUT / TARGET =================
# 🔥 decomposed Tmax level as predictor
X = df_temp[[level_name]]

# 🔥 raw ENSO index as target
y = df_enso[[enso_name]].values.ravel()

# ================= TRAIN-TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False, random_state=42
)

# ================= OBJECTIVE FUNCTION =================
def xgb_score(learning_rate, max_depth, subsample, colsample_bytree,
              gamma, min_child_weight, n_estimators):

    model = XGBRegressor(
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        min_child_weight=int(min_child_weight),
        n_estimators=int(n_estimators),
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return -mean_squared_error(y_test, y_pred)

# ================= PARAMETER BOUNDS =================
param_bounds = {
    'learning_rate': (0.001, 0.2),
    'max_depth': (3, 10),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'gamma': (0, 5),
    'min_child_weight': (1, 10),
    'n_estimators': (50, 200)
}

# ================= OPTIMIZATION =================
optimizer = BayesianOptimization(
    f=xgb_score,
    pbounds=param_bounds,
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=20)

# ================= BEST PARAMETERS =================
best_params = optimizer.max['params']

best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])
best_params['n_estimators'] = int(best_params['n_estimators'])

print("Best Hyperparameters:", best_params)

# ================= FINAL MODEL =================
best_model = XGBRegressor(
    **best_params,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

# ================= SAVE MODEL =================
model_name = f'{output_dir}XGB_{station_name}_{level_name}_{enso_name}.joblib'
dump(best_model, model_name)

# ================= PREDICTIONS =================
train_predict = best_model.predict(X_train)
test_predict = best_model.predict(X_test)

# ================= METRICS =================
train_r2 = r2_score(y_train, train_predict)
test_r2 = r2_score(y_test, test_predict)

train_MAE = mean_absolute_error(y_train, train_predict)
train_RMSE = np.sqrt(mean_squared_error(y_train, train_predict))
train_MSE = mean_squared_error(y_train, train_predict)

test_MAE = mean_absolute_error(y_test, test_predict)
test_RMSE = np.sqrt(mean_squared_error(y_test, test_predict))
test_MSE = mean_squared_error(y_test, test_predict)

# ================= SAVE RESULTS =================
df_metrics = pd.DataFrame({
    'Metric': ['R2', 'MAE', 'RMSE', 'MSE'],
    'Train': [train_r2, train_MAE, train_RMSE, train_MSE],
    'Test': [test_r2, test_MAE, test_RMSE, test_MSE]
})

df_predictions = pd.DataFrame({
    'y_test': y_test,
    'test_predict': test_predict
})

output_file = f'{output_dir}{station_name}_{level_name}_{enso_name}_{season}.xlsx'

with pd.ExcelWriter(output_file) as writer:
    df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
    df_predictions.to_excel(writer, sheet_name='Predictions', index=False)

print(f"Done! Results saved as {output_file}")