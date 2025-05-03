import os

class Config:
    # Veri yolları
    DATA_DIR = "data"
    TRAIN_DIR = os.path.join(DATA_DIR, "Training")
    TEST_DIR = os.path.join(DATA_DIR, "Test")
    
    # Model ayarları
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 64
    EPOCHS = 50
    INITIAL_EPOCHS = 15  # Dondurulmuş katmanlar için
    FINE_TUNE_EPOCHS = EPOCHS - INITIAL_EPOCHS
    
    # Model kayıt
    MODEL_DIR = "models"
    MODEL_NAME = "effnetb0_fruits360"
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
    
    # Feedback
    FEEDBACK_DIR = "feedback"
    FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback_data.csv")
    
    # Sınıf sayısı (otomatik belirlenecek)
    NUM_CLASSES = None
    CLASS_NAMES = None