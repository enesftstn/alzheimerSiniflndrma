from utils.data_loader import (
    get_class_names, 
    create_data_generators,
    get_class_distribution
)
from utils.model_utils import (
    build_model,
    compile_model,
    train_model,
    fine_tune_model,
    evaluate_model,
    plot_history
)
from config import Config
import pandas as pd
import os

def main():
    # Sınıf isimlerini ve sayısını al
    class_names = get_class_names()
    print(f"Toplam {Config.NUM_CLASSES} sınıf bulundu:")
    print(class_names[:10], "...")  # İlk 10 sınıfı göster

    # Veri yükleyicileri oluştur
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Sınıf dağılımını analiz et
    class_dist = get_class_distribution(train_gen)
    print("\nSınıf Dağılımı:")
    print(class_dist.head(10))  # İlk 10 sınıfın dağılımı
    
    # Modeli oluştur ve derle
    model, base_model = build_model(Config.NUM_CLASSES)
    model = compile_model(model)
    model.summary()

    # Modeli eğit
    print("\nİlk eğitim aşaması (dondurulmuş katmanlar)...")
    model, history = train_model(model, train_gen, val_gen)
    
    # Fine-tuning
    print("\nFine-tuning aşaması...")
    model, fine_history = fine_tune_model(model, base_model, train_gen, val_gen)
    
    # Modeli değerlendir
    print("\nModel değerlendirme...")
    class_acc = evaluate_model(model, test_gen)
    
    # Eğitim geçmişini görselleştir
    plot_history(history, fine_history)
    
    # En iyi ve en kötü performans gösteren sınıflar
    print("\nEn iyi performans gösteren 10 sınıf:")
    print(class_acc.nlargest(10, 'accuracy'))
    
    print("\nEn kötü performans gösteren 10 sınıf:")
    print(class_acc.nsmallest(10, 'accuracy'))
    
    # Modeli kaydet
    model.save(Config.MODEL_PATH)
    print(f"\nModel kaydedildi: {Config.MODEL_PATH}")

if __name__ == "__main__":
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    main()