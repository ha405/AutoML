import os
import requests
import argparse

parser = argparse.ArgumentParser(description="Test DataWise AI Pipeline")
parser.add_argument("--host", default="http://127.0.0.1:5000", help="Backend host URL")
args = parser.parse_args()

s = requests.Session()
base_url = args.host.rstrip('/')

print("1. Hitting /api/home")
with open("TestDatasets/CarPrice_Assignment.csv", "rb") as f:
    res = s.post(f"{base_url}/api/home", data={"queryInput": "I need to analyze this data."}, files={"file-upload": f})
print("Status:", res.status_code)

print("\n2. Hitting /api/conversation")
res = s.post(f"{base_url}/api/conversation", json={"user_response": "My final goal is to predict the price of a car using a regression model. I need to know which features correlate the most with the price."})
print("Status:", res.status_code)
data = res.json()
if not data.get("final_problem_determined"):
    print("Not determined yet, forcing finalization...")
    res = s.post(f"{base_url}/api/conversation", json={"user_response": "Yes, exactly. Let's proceed with that."})
    print("Status:", res.status_code)

print("\n3. Hitting /api/superllm")
res = s.post(f"{base_url}/api/superllm", timeout=120)
print("Status:", res.status_code)
if res.status_code != 200:
    print(res.text)

print("\n4. Hitting /api/dataanalysis")
res = s.post(f"{base_url}/api/dataanalysis", timeout=300)
print("Status:", res.status_code)
if res.status_code != 200:
    print(res.text)

print("\n5. Hitting /api/ml")
res = s.post(f"{base_url}/api/ml", timeout=300)
print("Status:", res.status_code)
if res.status_code != 200:
    print(res.text)

print("\n6. Hitting /api/vlm")
res = s.post(f"{base_url}/api/vlm", timeout=120)
print("Status:", res.status_code)
if res.status_code != 200:
    print(res.text)

print("\nFull pipeline test completed successfully.")
