# FastAPI Service for Real Estate Predictions

## Description
This FastAPI service accepts CSV files with real estate data and returns predictions for rent and sale prices, along with model evaluation metrics.

## Installation and Setup

### Step 1: Clone the repository
sh
git clone <repository_url>
cd <repository_name>


### Step 2: Create a virtual environment and install dependencies

python3 -m venv venv
source venv/bin/activate
pip install -r [requirements.txt](requirements.txt)

### Step 3: Run the service (main.py)[main.py]
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

### Step 4: Build and run the Docker container (dockerfile)[dockerfile]
docker build -t my_fastapi_service .
docker run -d -p 8000:8000 my_fastapi_service

## Usage
### Rent Prediction
Send a POST request to the /predict_rent endpoint with a CSV file:

curl -X POST "http://localhost:8000/predict_rent" -H "Content-Type: multipart/form-data" -F "file=@path_to_your_file.csv"

### Sale Price Prediction
Send a POST request to the /predict_trans endpoint with a CSV file:
curl -X POST "http://localhost:8000/predict_trans" -H "Content-Type: multipart/form-data" -F "file=@path_to_your_file.csv"

## Testing
### Step 1: Run the tests
python (test.py)[test.py].



