import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
import os

MODELS_PATH = "models"
DATA_PATH = "data"


train_df = pd.read_csv(os.path.join(DATA_PATH, "jobfair_train.csv"))

train_df["global_competition_level"].fillna(0, inplace=True)

train_df.sort_values(by="league_id", inplace=True)

train_df.reset_index(inplace=True, drop=True)

columns_to_drop = ["season", "club_id", "league_id", "registration_country", "registration_platform_specific"]
train_df.drop(columns_to_drop, axis=1, inplace=True)

encoding_map = {
    '0) NonPayer': 0,
    '1) ExPayer': 1,
    '2) Minnow': 2,
    '3) Dolphin': 3,
    '4) Whale': 4
}
train_df["dynamic_payment_segment"] = train_df["dynamic_payment_segment"].map(encoding_map)

np.random.seed(42)

num_chunks = len(train_df) // 14

league_order = np.arange(num_chunks)

np.random.shuffle(league_order)

shuffled_df = pd.concat([train_df.iloc[x * 14: (x + 1) * 14] for x in league_order])

shuffled_df.reset_index(drop=True, inplace=True)

train_percentage = 0.8
num_train_rows = int(len(shuffled_df) * train_percentage)

num_train_chunks = num_train_rows // 14 * 14

train_df = shuffled_df.iloc[:num_train_chunks]
test_df = shuffled_df.iloc[num_train_chunks:]

X_train = train_df.drop("league_rank", axis=1)
y_train = train_df["league_rank"]

X_valid = test_df.drop("league_rank", axis=1)
y_valid = test_df["league_rank"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

xgb_params = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'n_estimators': 300
}

xgbr_model = XGBRegressor(**xgb_params)


xgbr_model.fit(X_train_scaled, y_train)

y_pred = xgbr_model.predict(X_valid_scaled)

mae = mean_absolute_error(y_valid, y_pred)

joblib.dump(xgbr_model, os.path.join(MODELS_PATH, "xgbr_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))

print("------------------------------------")
print(f"XgboostRegressor trained with MAE: {mae}")
print("------------------------------------")
