import requests
def test_predict():
    url = 'http://localhost:8000/predict'
    payload = {'input': 'Hello world', 'max_new_tokens': 20}
    r = requests.post(url, json=payload, timeout=10)
    print('Status:', r.status_code)
    print('Response:', r.json())
if __name__=='__main__':
    test_predict()