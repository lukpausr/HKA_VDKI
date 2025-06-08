import os
import time
from datetime import datetime
import torch
import pytorch_lightning as pl
from torchvision.transforms import v2

from config.load_configuration import load_configuration

# disable oneDNN optimizations for reproducibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# THREAD (läuft auf SteuerungsPC)
# --> Dauerschleife

from camera import Camera
from rest_api import RabbitAPIClient
from data.custom_transforms import CenterCropSquare
from models.model_facedetection import FD_ConvNextV2
from models.model_transferlearning import TL_ConvNextV2

class RabbitRecognitionSystem:
    def __init__(self):
        # Konfiguration laden
        self.config = load_configuration()

        # RestAPI Client importieren und initialisieren
        self.api_client = RabbitAPIClient(base_url=self.config['rest_api_base_url'])
        ""
        # Kamera initialisieren
        self.camera = Camera(camera_id=0)
        self.camera.connect()
        self.camera.configure_camera(resolution=(640, 480), fps=30)

        # Modelle initialisieren
        self.model = TL_ConvNextV2.load_from_checkpoint(self.config['path_to_tl_model'], map_location=device)
        self.model_facedetection = FD_ConvNextV2.load_from_checkpoint(self.config['path_to_fd_model'], map_location=device)

        self.model.eval()
        self.model_facedetection.eval()

        # Transformationen definieren
        self.transform = v2.Compose([
            CenterCropSquare(),
            v2.Resize((self.config['image_size'], self.config['image_size'])),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run(self):
        while True:
            # Licht einschalten
            self.api_client.turn_light_on()
            while not self.api_client.is_light_on():
                time.sleep(0.1)
            image = self.camera.get_image()
            self.api_client.turn_light_off()
            while self.api_client.is_light_on():
                time.sleep(0.1)
            # Bild vorverarbeiten
            image_transformed = self.transform(image)
            # Modellvorhersage
            prediction = self.model(image_transformed.unsqueeze(0))
            if prediction.item() == 1:
                prediction_facedetection = self.model_facedetection(image_transformed.unsqueeze(0))
            else:
                prediction_facedetection = None
            # REST-API POST
            response = self.api_client.post_animalRecognition_data(
                prediction=prediction.item(),
                prediction_facedetection=prediction_facedetection.item() if prediction_facedetection is not None else None
            )
            if response is not None:
                print("POST request successful:", response.json())
            else:
                print("POST request failed. No action taken.")
            # Bild speichern
            if prediction.item() == 1:
                image_path = os.path.join('images', 'kaninchen', f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path)
            # Healthcheck
            if not self.api_client.health_check():
                raise ConnectionError("REST-API nicht erreichbar. Bitte überprüfen Sie die Verbindung.")
            else:
                print("REST-API erreichbar.")
            time.sleep(5)

if __name__ == "__main__":
    visionSystem = RabbitRecognitionSystem()
    visionSystem.run()
