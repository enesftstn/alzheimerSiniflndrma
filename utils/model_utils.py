import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)
from config import Config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os

def build_model(num_classes):
    """EfficientNetB0 tabanlı model oluşturur"""
    # Temel model (dondurulmuş)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*Config.IMG_SIZE, 3),
        pooling='avg'
    )
    base_model.trainable = False

    # Yeni model
    inputs = tf.keras.Input(shape=(*Config.IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    
    return model, base_model

def compile_model(model):
    """Modeli derler"""
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model

def train_model(model, train_generator, val_generator):
    """Modeli eğitir"""
    # Callback'ler
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            Config.MODEL_PATH,
            save_best_only=True,
            save_weights_only=False
        ),
        ReduceLROnPlateau(factor=0.2, patience=3),
        TensorBoard(log_dir='logs')
    ]

    # İlk eğitim (dondurulmuş katmanlar)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // Config.BATCH_SIZE,
        epochs=Config.INITIAL_EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // Config.BATCH_SIZE,
        callbacks=callbacks
    )

    return model, history

def fine_tune_model(model, base_model, train_generator, val_generator):
    """Modeli fine-tune eder"""
    # Bazı katmanları çözme
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Düşük learning rate ile tekrar derleme
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fine-tuning
    fine_history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        initial_epoch=Config.INITIAL_EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // Config.BATCH_SIZE,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(Config.MODEL_PATH, save_best_only=True)
        ]
    )

    return model, fine_history

def evaluate_model(model, test_generator):
    """Modeli değerlendirir"""
    # Test verisi üzerinde değerlendirme
    results = model.evaluate(test_generator)
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")

    # Sınıflandırma raporu
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(
        test_generator.classes, 
        y_pred_classes,
        target_names=test_generator.class_indices.keys()
    ))

    # Confusion matrix
    cm = confusion_matrix(test_generator.classes, y_pred_classes)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_generator.class_indices.keys(), 
                yticklabels=test_generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Sınıf başına doğruluk
    class_acc = {}
    for i, class_name in enumerate(test_generator.class_indices.keys()):
        mask = test_generator.classes == i
        class_acc[class_name] = np.mean(y_pred_classes[mask] == i)
    
    return pd.DataFrame.from_dict(class_acc, orient='index', columns=['accuracy'])

def plot_history(history, fine_history=None):
    """Eğitim geçmişini görselleştirir"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    if fine_history:
        acc += fine_history.history['accuracy']
        val_acc += fine_history.history['val_accuracy']
        loss += fine_history.history['loss']
        val_loss += fine_history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()