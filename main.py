from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from src.rent_model import predict_rent, get_rent_model_features, get_rent_model_feature_types
from src.trans_model import predict_trans, get_trans_model_features, get_trans_model_feature_types
from src.config import TARGET_COLUMN_RENTS, TARGET_COLUMN_TRANS
from src.evaluation import evaluate_model
app = FastAPI()

# Получение списка фичей и их типов для моделей
rent_features = get_rent_model_features()
rent_feature_types = get_rent_model_feature_types()
trans_features = get_trans_model_features()
trans_feature_types = get_trans_model_feature_types()

# Определение типов данных для Pydantic моделей
def get_pydantic_type(catboost_type):
    if catboost_type == 'float':
        return (float, ...)
    elif catboost_type == 'int':
        return (int, ...)
    else:
        return (str, ...)

# Создание динамических моделей для запросов
RentRequest = create_model('RentRequest', **{feature: get_pydantic_type(rent_feature_types[i]) for i, feature in enumerate(rent_features)})
TransRequest = create_model('TransRequest', **{feature: get_pydantic_type(trans_feature_types[i]) for i, feature in enumerate(trans_features)})

@app.post("/predict_rent")
def predict_rent_endpoint(request: RentRequest):
    try:
        prediction = predict_rent(request.dict())
        return {"rent_prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_trans")
def predict_trans_endpoint(request: TransRequest):
    try:
        prediction = predict_trans(request.dict())
        return {"trans_prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")