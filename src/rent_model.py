from catboost import CatBoostRegressor
from src.config import NAME_RENTS_MODEL

rent_model = CatBoostRegressor()
rent_model.load_model(f"models/{NAME_RENTS_MODEL}")

def get_rent_model_features():
    return rent_model.feature_names_

def get_rent_model_feature_types():
    return rent_model.get_feature_importance(prettified=True)['Feature Id'].tolist()

def predict_rent(data):
    features = [data[feature] for feature in rent_model.feature_names_]
    prediction = rent_model.predict([features])
    return prediction[0]