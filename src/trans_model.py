from catboost import CatBoostRegressor
from src.config import NAME_TRANS_MODEL
trans_model = CatBoostRegressor()
trans_model.load_model(f"models/{NAME_TRANS_MODEL}")

def get_trans_model_features():
    return trans_model.feature_names_

def get_trans_model_feature_types():
    return trans_model.get_feature_importance(prettified=True)['Feature Id'].tolist()

def predict_trans(data):
    features = [data[feature] for feature in trans_model.feature_names_]
    prediction = trans_model.predict([features])
    return prediction[0]