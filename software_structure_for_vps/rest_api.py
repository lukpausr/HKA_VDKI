import requests
from datetime import datetime

class RabbitAPIClient:
    def __init__(self, base_url='http://localhost:8000'):
        self.base_url = base_url.rstrip('/')

    def post_animalRecognition_data(self, prediction, prediction_facedetection):
        url = f"{self.base_url}/your_endpoint"  # Adjust endpoint path if needed
        payload = {
            'animal_type': prediction,                  # Assuming prediction is a string like 'rabbit' or 'not_rabbit'
            'rabbit_id': prediction_facedetection,      # Assuming this is an identifier for the rabbit
            'timestamp': datetime.now().isoformat()
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise error for bad status codes
            print("POST successful:", response.json())
            return response
        except requests.exceptions.RequestException as e:
            print("POST failed:", e)
            return None

        def health_check(self):
                url = f"{self.base_url}/health"
                try:
                    response = requests.get(url)
                    return response.status_code == 200
                except requests.exceptions.RequestException:
                    return False