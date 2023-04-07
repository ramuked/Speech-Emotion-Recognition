import os

sample_rate = 16000  # sample rate
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # project path

SAVED_MODEL_PATH = "/home/sushant/Desktop/Sem7/MajorProjectSem7/django-app/SER/homepage/sermodel/model"  # path to save trained model

SAVED_WEIGHTS_PATH = "/home/sushant/Desktop/Sem7/MajorProjectSem7/django-app/SER/homepage/sermodel/checkpoints/my_checkpoint"


SAVED_SCALER_PATH = "/home/sushant/Desktop/Sem7/MajorProjectSem7/django-app/SER/homepage/sermodel/sca1.gz"

SAVED_ENCODER_PATH = "/home/sushant/Desktop/Sem7/MajorProjectSem7/django-app/SER/homepage/sermodel/enc1.gz"
MEDIA_DIR = os.path.abspath(__file__ + 3 * "/.." + "/media")  # path to media folder
