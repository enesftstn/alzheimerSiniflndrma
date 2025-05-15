from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from config import Config

def build_model(num_classes):
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_gen, val_gen):
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            Config.MODEL_PATH,
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=val_gen,
        validation_steps=val_gen.samples // Config.BATCH_SIZE,
        callbacks=callbacks
    )
    
    return model, history

def evaluate_model(model, test_gen):
    results = model.evaluate(test_gen)
    print(f"Test Accuracy: {results[1]:.4f}")
    
    # Confusion matrix
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Sınıf başına doğruluk
    class_acc = {}
    for i, class_name in enumerate(test_gen.class_indices.keys()):
        mask = test_gen.classes == i
        class_acc[class_name] = np.mean(y_pred_classes[mask] == i)
    
    return pd.DataFrame.from_dict(class_acc, orient='index', columns=['accuracy'])
