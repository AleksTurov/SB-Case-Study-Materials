import requests
import gdown
import pandas as pd


# dowload csv files
def test_download_csv():
    # URL to the file on Google Drive
    url_rents = 'https://drive.google.com/uc?id=1oc_RJRsQEiJutVdWjlRdbJ773YpE9h6x'
    output_rents = 'data/snp_dld_2024_rents.csv'
    url_trans = 'https://drive.google.com/uc?id=1liNykIOnfR5KRR4MXJISCZCKJYFnXQC1'
    output_trans = 'data/snp_dld_2024_transactions.csv'
    # Download the file
    gdown.download(url_rents, output_rents, quiet=False)
    gdown.download(url_trans, output_trans, quiet=False)

    # Load data into DataFrame
    snp_dld_2024_rents = pd.read_csv(output_rents)
    snp_dld_2024_transactions = pd.read_csv(output_trans)
    print(snp_dld_2024_rents.shape)
    print(snp_dld_2024_transactions.shape)


# Test the prediction API
def test_predict_rent():
    url = "http://localhost:8000/predict_rent"
    files = {'file': open('data/snp_dld_2024_rents.csv', 'rb')}
    response = requests.post(url, files=files)
    assert response.status_code == 200
    print("Rent prediction test passed")
    print("Predictions:", response.json())

def test_predict_trans():
    url = "http://localhost:8000/predict_trans"
    files = {'file': open('data/snp_dld_2024_transactions.csv', 'rb')}
    response = requests.post(url, files=files)
    assert response.status_code == 200
    print("Transaction prediction test passed")
    print("Predictions:", response.json())

if __name__ == "__main__":
    test_download_csv()
    test_predict_rent()
    #test_predict_trans()