import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2  # ✅ Faster model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ✅ Enable XLA Acceleration (Speeds up CPU training)
tf.config.optimizer.set_jit(True)

# ✅ Get the base directory dynamically
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "food-101", "food-101", "images")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ ERROR: Dataset path not found: {DATA_DIR}")
else:
    print(f"✅ Dataset found at: {DATA_DIR}")

# ✅ Optimize image processing for faster CPU training
IMG_SIZE = (128, 128)  # Lower resolution for speed
BATCH_SIZE = 16  # Smaller batch size for CPU efficiency

# ✅ Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,  # Reduced for speed
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ✅ Load training and validation data
train_generator = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", subset="training"
)
validation_generator = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", subset="validation"
)

# ✅ Load MobileNetV2 (Faster than EfficientNet)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze for transfer learning

# ✅ Define a light model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(101, activation="softmax")  # 101 food categories
])

# ✅ Compile for fast execution
model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Early stopping to prevent overtraining
early_stopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

# ✅ Train model with optimizations
history = model.fit(
    train_generator,
    epochs=3,  # Fewer epochs
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# ✅ Save trained model
MODEL_PATH = os.path.join(BASE_DIR, "food_recognition_model.h5")
model.save(MODEL_PATH)

print(f" Model training complete and saved at {MODEL_PATH}")
