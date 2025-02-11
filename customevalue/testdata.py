import requests

url = "http://127.0.0.1:5000/predict"

# Correct JSON structure
data = {
    "Pclass": 3,       # Class
    "Sex": "male",     # Can be "male" or "female"
    "Age": 25,         # Age
    "Fare": 7.25       # Ticket fare
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)

# Debugging: Print raw response
print("Status Code:", response.status_code)
print("Raw Response:", response.text)

# Try parsing JSON
try:
    print("JSON Response:", response.json())
except requests.exceptions.JSONDecodeError:
    print("Error: Response is not valid JSON")
