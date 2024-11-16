import requests

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
    test_predict_rent()
    #test_predict_trans()