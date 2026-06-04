import requests

TEACHER_BASE_URL = "http://192.168.50.218:8000/api/v1"
STUDENT_SERVER_URL = "http://192.168.50.154:5000"
STUDENT_ID = "B22DCAT082"

headers = {
    "X-Student-ID": STUDENT_ID
}

def register():
    url = f"{TEACHER_BASE_URL}/competition/register"
    payload = {"server_url": STUDENT_SERVER_URL}
    res = requests.post(url, headers=headers, json=payload)
    print("Register Response:", res.json())

def evaluate():
    url = f"{TEACHER_BASE_URL}/competition/evaluate"
    res = requests.post(url, headers=headers)
    print("Evaluate Response:", res.json())
    
def reset():
    url = f"{TEACHER_BASE_URL}/competition/reset"
    res = requests.post(url, headers=headers)
    print("Reset Response:", res.json())

def result():
    url = f"{TEACHER_BASE_URL}/competition/result"
    res = requests.get(url, headers=headers)
    print("Result Response:", res.json())

if __name__ == "__main__":
    reset()
    register()
    evaluate()
    result()