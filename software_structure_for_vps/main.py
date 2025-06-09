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
    class Zustand:
        INITIALISIERUNG = "INITIALISIERUNG"
        BEREIT = "BEREIT"
        BILDAUFNAHME = "BILDAUFNAHME"
        ZK_ERKENNUNG = "ZK_ERKENNUNG"  # Erkennung, ob ein Zwergkaninchen vor der Türe steht
        ZK_UNTERSCHEIDUNG = "ZK_UNTERSCHEIDUNG"  # Kann Zwergkaninchen voneinander unterscheiden
        API_POST_ZK = "API_POST_ZK"
        API_GESUNDHEITSPRUEFUNG = "API_GESUNDHEITSPRUEFUNG"

    def __init__(self):
        self.zustand = self.Zustand.INITIALISIERUNG
        self.config = None
        self.api_client = None
        self.kamera = None
        self.zk_kennung_model = None
        self.zk_unterscheidung_model = None
        self.transform = None
        self.bild = None
        self.bild_transformiert = None
        self.zk_kennung_vorhersage = None
        self.zk_unterscheidung_vorhersage = None
        self.response = None

    def run(self):
        while True:
            if self.zustand == self.Zustand.INITIALISIERUNG:
                # Konfiguration laden
                self.config = load_configuration()
                # API-Client initialisieren
                self.api_client = RabbitAPIClient(base_url=self.config['rest_api_base_url'])
                # Kamera initialisieren
                self.kamera = Camera(camera_id=0)
                self.kamera.connect()
                self.kamera.configure_camera(resolution=(640, 480), fps=30)
                # Modelle initialisieren
                self.zk_kennung_model = TL_ConvNextV2.load_from_checkpoint(self.config['path_to_tl_model'], map_location=device)
                self.zk_unterscheidung_model = FD_ConvNextV2.load_from_checkpoint(self.config['path_to_fd_model'], map_location=device)
                self.zk_kennung_model.eval()
                self.zk_unterscheidung_model.eval()
                # Transforms definieren
                self.transform = v2.Compose([
                    CenterCropSquare(),
                    v2.Resize((self.config['image_size'], self.config['image_size'])),
                    v2.ToTensor(),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self.zustand = self.Zustand.BEREIT

            elif self.zustand == self.Zustand.BEREIT:
                time.sleep(5)
                self.zustand = self.Zustand.BILDAUFNAHME

            elif self.zustand == self.Zustand.BILDAUFNAHME:
                self.bild = self.kamera.get_image()
                self.zustand = self.Zustand.ZK_ERKENNUNG

            elif self.zustand == self.Zustand.ZK_ERKENNUNG:
                self.bild_transformiert = self.transform(self.bild)
                self.zk_kennung_vorhersage = self.zk_kennung_model(self.bild_transformiert.unsqueeze(0))
                if self.zk_kennung_vorhersage.item() == 1:
                    self.zustand = self.Zustand.ZK_UNTERSCHEIDUNG
                else:
                    self.zk_unterscheidung_vorhersage = None
                    self.zustand = self.Zustand.BEREIT

            elif self.zustand == self.Zustand.ZK_UNTERSCHEIDUNG:
                # Zwergkaninchen-Unterscheidung kann Zwergkaninchen voneinander unterscheiden
                self.zk_unterscheidung_vorhersage = self.zk_unterscheidung_model(self.bild_transformiert.unsqueeze(0))
                self.zustand = self.Zustand.API_POST_ZK

            elif self.zustand == self.Zustand.API_POST_ZK:
                self.response = self.api_client.post_animalRecognition_data(
                    prediction=self.zk_kennung_vorhersage.item(),
                    prediction_facedetection=self.zk_unterscheidung_vorhersage.item() if self.zk_unterscheidung_vorhersage is not None else None
                )
                if self.response is not None:
                    print("POST request erfolgreich:", self.response.json())
                    # Bild speichern, wenn Tier erkannt wurde
                    if self.zk_kennung_vorhersage.item() == 1:
                        image_path = os.path.join('images', 'kaninchen', f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                        os.makedirs(os.path.dirname(image_path), exist_ok=True)
                        self.bild.save(image_path)
                    self.zustand = self.Zustand.BEREIT
                else:
                    print("POST request fehlgeschlagen. Keine Aktion durchgeführt.")
                    self.zustand = self.Zustand.API_GESUNDHEITSPRUEFUNG

            elif self.zustand == self.Zustand.API_GESUNDHEITSPRUEFUNG:
                while not self.api_client.health_check():
                    print("REST-API nicht erreichbar. Bitte überprüfen Sie die Verbindung.")
                    time.sleep(5)
                print("REST-API erreichbar.")
                self.zustand = self.Zustand.BEREIT

if __name__ == "__main__":
    visionSystem = RabbitRecognitionSystem()
    visionSystem.run()
