import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Set parameters
data_dir = 'mini_train'
img_size = 64

X, y, labels = [], [], []

print("📦 Loading images...")
class_folders = sorted(os.listdir(data_dir))

for idx, cls in enumerate(class_folders):
    folder_path = os.path.join(data_dir, cls)
    if not os.path.isdir(folder_path):
        continue
    labels.append(cls)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(idx)

# Prepare data
X = np.array(X) / 255.0
y = to_categorical(y, num_classes=len(labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🏋️ Training model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save('asl_model.h5')
print("✅ Model saved as 'asl_model.h5'")
print("🔤 Labels:", labels)
