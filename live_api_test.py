"""
Script to test the live API deployment.

This script sends a POST request to the deployed API and prints
both the status code and the prediction result.

Usage:
    python live_api_test.py

Before running, update the API_URL variable with your deployed API URL.
"""

import requests
import json


# Update this with your deployed Heroku app URL
API_URL = "https://census-income-classifier.onrender.com/"


def test_get_root():
    """Test GET request on root endpoint."""
    print("=" * 60)
    print("Testing GET /")
    print("=" * 60)

    response = requests.get(f"{API_URL}/")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

    return response.status_code, response.json()


def test_post_predict_low_income():
    """Test POST request for low income prediction."""
    print("=" * 60)
    print("Testing POST /predict (Expected: <=50K)")
    print("=" * 60)

    # Sample data for a person likely earning <=50K
    sample_data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }

    print(f"Input Data: {json.dumps(sample_data, indent=2)}")
    print()

    response = requests.post(
        f"{API_URL}/predict",
        json=sample_data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

    return response.status_code, response.json()


def test_post_predict_high_income():
    """Test POST request for high income prediction."""
    print("=" * 60)
    print("Testing POST /predict (Expected: >50K)")
    print("=" * 60)

    # Sample data for a person likely earning >50K
    sample_data = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }

    print(f"Input Data: {json.dumps(sample_data, indent=2)}")
    print()

    response = requests.post(
        f"{API_URL}/predict",
        json=sample_data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

    return response.status_code, response.json()


def main():
    """Run all live API tests."""
    print("\n" + "=" * 60)
    print("LIVE API TEST - Census Income Classification")
    print(f"API URL: {API_URL}")
    print("=" * 60 + "\n")

    try:
        # Test GET /
        get_status, get_response = test_get_root()

        # Test POST /predict for <=50K
        post_low_status, post_low_response = test_post_predict_low_income()

        # Test POST /predict for >50K
        post_high_status, post_high_response = test_post_predict_high_income()

        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"GET /: Status {get_status} - {'PASS' if get_status == 200 else 'FAIL'}")
        print(f"POST /predict (<=50K): Status {post_low_status} - "
              f"{'PASS' if post_low_status == 200 else 'FAIL'}")
        print(f"POST /predict (>50K): Status {post_high_status} - "
              f"{'PASS' if post_high_status == 200 else 'FAIL'}")

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API.")
        print(f"Please verify that {API_URL} is correct and the server is running.")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
