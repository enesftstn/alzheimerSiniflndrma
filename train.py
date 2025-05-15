from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from utils.data_loader import create_data_generators, get_class_distribution
from utils.model_utils import build_model, train_model, evaluate_model
from config import Config
import os
import pandas as pd
def main():
    # Veri yükleyicileri oluştur
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Sınıf dağılımını görselleştir
    class_dist = get_class_distribution(Config.TRAIN_DIR)
    print("Sınıf Dağılımı:\n", class_dist)
    
    # Modeli oluştur
    model = build_model(Config.NUM_CLASSES)
    
    # Modeli eğit
    print("\nModel eğitimi başlıyor...")
    model, history = train_model(model, train_gen, val_gen)
    
    # Modeli değerlendir
    print("\nModel değerlendirme...")
    results = evaluate_model(model, test_gen)
    
    # Modeli kaydet
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    model.save(Config.MODEL_PATH)
    print(f"\nModel kaydedildi: {Config.MODEL_PATH}")

if __name__ == "__main__":
    main()
