import os

class Config:
    # Veri yolları
    DATA_DIR = "data"
    TRAIN_DIR = os.path.join(DATA_DIR, "Training")
    TEST_DIR = os.path.join(DATA_DIR, "Test")
    
    # Model ayarları
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    NUM_CLASSES = 4  # 4 farklı sınıf
    
    # Model kayıt
    MODEL_DIR = "models"
    MODEL_NAME = "alzheimer_model.h5"
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
    
    @classmethod
    def check_paths(cls):
        """Gerekli klasörleri kontrol eder ve oluşturur"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        if not os.path.exists(cls.MODEL_PATH):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {cls.MODEL_PATH}")
