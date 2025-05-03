import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import Config
import pandas as pd
import os

def get_class_names(data_dir=Config.TRAIN_DIR):
    """Sınıf isimlerini ve sayısını döndürür"""
    class_names = sorted(os.listdir(data_dir))
    Config.NUM_CLASSES = len(class_names)
    Config.CLASS_NAMES = class_names
    return class_names

def create_data_generators():
    """Veri artırma ve yükleme işlemleri"""
    # Eğitim için augmentasyon
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.1  # Doğrulama için %10 ayır
    )

    # Test ve validasyon için sadece normalizasyon
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Eğitim verisi
    train_generator = train_datagen.flow_from_directory(
        Config.TRAIN_DIR,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validasyon verisi
    val_generator = train_datagen.flow_from_directory(
        Config.TRAIN_DIR,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    # Test verisi
    test_generator = test_datagen.flow_from_directory(
        Config.TEST_DIR,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def get_class_distribution(generator):
    """Sınıf dağılımını analiz eder"""
    class_counts = {}
    for cls, idx in generator.class_indices.items():
        class_counts[cls] = sum(generator.classes == idx)
    
    return pd.DataFrame.from_dict(class_counts, orient='index', columns=['count'])