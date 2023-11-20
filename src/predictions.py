import pandas as pd
import joblib

test_df = pd.read_csv("data/jobfair_test.csv")

test_df["global_competition_level"].fillna(0, inplace=True)

test_df.sort_values(by="league_id", inplace=True)

test_df.reset_index(inplace=True, drop=True)

columns_to_drop = ["season", "league_id", "registration_country", "registration_platform_specific"]
test_df.drop(columns_to_drop, axis=1, inplace=True)

encoding_map = {
    '0) NonPayer': 0,
    '1) ExPayer': 1,
    '2) Minnow': 2,
    '3) Dolphin': 3,
    '4) Whale': 4
}
test_df["dynamic_payment_segment"] = test_df["dynamic_payment_segment"].map(encoding_map)


model = joblib.load("models/xgbr_model.pkl")
scaler = joblib.load("models/scaler.pkl")


rank_predictions = {}

for i in range(0, len(test_df), 14):
    league_df = test_df[i:i + 14].copy()
    
    club_ids = list(league_df["club_id"])
    league_df.drop("club_id", axis=1, inplace=True)
    
    scaled_data = scaler.transform(league_df)
    predictions = model.predict(scaled_data)
    
    combined = list(zip(club_ids, predictions))
    sorted_combined = sorted(combined, key=lambda x: x[1])
    
    sorted_club_ids = [pair[0] for pair in sorted_combined]
    rank_preds = [x for x in range(1, 15)]

    for id, rank in zip(sorted_club_ids, rank_preds):
        rank_predictions[id] = rank
        
        
results_df = pd.DataFrame(list(rank_predictions.items()), columns=['club_id', 'league_rank'])

results_df.to_csv("league_rank_predictions.csv", index=False)

print("PREDICTIONS ARE SAVED!")
