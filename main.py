from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, create_model
from src.rent_model import predict_rent, get_rent_model_features, get_rent_model_feature_types
from src.trans_model import predict_trans, get_trans_model_features, get_trans_model_feature_types
from src.config import TARGET_COLUMN_RENTS, TARGET_COLUMN_TRANS
from src.evaluation import evaluate_model
from src.preprocess import preprocess_data_rents, preprocess_data_trans
import pandas as pd
import io
import numpy as np

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
async def predict_rent_from_csv(file: UploadFile = File(...)):
    try:
        # Чтение CSV файла
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), low_memory=False).head(100)
        
        # Предобработка данных
        df = preprocess_data_rents(df)
        
        # Проверка формата данных
        missing_features = [feature for feature in rent_features if feature not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing_features)}")
        
        # Проверка наличия целевой переменной
        if TARGET_COLUMN_RENTS not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing target column: {TARGET_COLUMN_RENTS}")
        
        # Обработка NaN и бесконечных значений
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        # Предсказания для аренды
        rent_predictions = df.apply(lambda row: predict_rent(row.to_dict()), axis=1)
        df['rent_prediction'] = rent_predictions
        
        # Оценка модели
        rent_metrics = evaluate_model(df[TARGET_COLUMN_RENTS], df['rent_prediction'])
        
        # Обработка NaN и бесконечных значений в предсказаниях
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        return {
            "rent_metrics": rent_metrics,
            "predictions": df.to_dict(orient='records'),
            "rent_metrics": rent_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_trans")
async def predict_trans_from_csv(file: UploadFile = File(...)):
    try:
        # Чтение CSV файла
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), low_memory=False).head(100)
        
        # Предобработка данных
        df = preprocess_data_trans(df)
        
        # Проверка формата данных
        missing_features = [feature for feature in trans_features if feature not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing_features)}")
        
        # Проверка наличия целевой переменной
        if TARGET_COLUMN_TRANS not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing target column: {TARGET_COLUMN_TRANS}")
        
        # Обработка NaN и бесконечных значений
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        # Предсказания для продажи
        trans_predictions = df.apply(lambda row: predict_trans(row.to_dict()), axis=1)
        df['trans_prediction'] = trans_predictions
        
        # Оценка модели
        trans_metrics = evaluate_model(df[TARGET_COLUMN_TRANS], df['trans_prediction'])
        
        # Обработка NaN и бесконечных значений в предсказаниях
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        return {
            "trans_metrics": trans_metrics,
            "predictions": df.to_dict(orient='records'),
            "trans_metrics": trans_metrics

        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")