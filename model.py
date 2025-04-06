import random
import numpy as np
import tensorflow as tf
import os
import json
from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
 
# Load Kaggle credentials
kaggle_credentials = json.load(open("kaggle.json"))
os.environ['KAGGLE_USERNAME'] = kaggle_credentials["username"]
os.environ['KAGGLE_KEY'] = kaggle_credentials["key"]

# Download and extract dataset
#os.system("kaggle datasets download -d abdallahalidev/plantvillage-dataset")
#with ZipFile("plantvillage-dataset.zip", 'r') as zip_ref:
 #   zip_ref.extractall()
 # Download and extract dataset
zip_path = "plantvillage-dataset.zip"

# Download only if the file doesn't already exist
if not os.path.exists(zip_path):
    download_status = os.system("kaggle datasets download -d abdallahalidev/plantvillage-dataset")
    if download_status != 0:
        raise Exception("Dataset download failed! Check Kaggle CLI, credentials, or internet connection.")

# Extract dataset if download was successful
if os.path.exists(zip_path):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
else:
    raise FileNotFoundError("The dataset zip file was not found even after attempting download.")


# Dataset Path
base_dir = 'plantvillage dataset/color'
img_size = 224
batch_size = 32

# Image Data Generators
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = data_gen.flow_from_directory(
    base_dir, target_size=(img_size, img_size), batch_size=batch_size, subset='training', class_mode='categorical'
)
validation_generator = data_gen.flow_from_directory(
    base_dir, target_size=(img_size, img_size), batch_size=batch_size, subset='validation', class_mode='categorical'
)

# Model Definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Model Evaluation
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the Model
model.save('conv2d.h5')

# Function to Load and Preprocess an Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.array(img, dtype='float32') / 255.0
    return np.expand_dims(img_array, axis=0)

# Function to Predict Image Class
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    return class_indices[np.argmax(predictions)]

# Save Class Indices
class_indices = {v: k for k, v in train_generator.class_indices.items()}
json.dump(class_indices, open('class_indices.json', 'w'))

# Load and Evaluate Model
loaded_model = tf.keras.models.load_model('conv2d.h5')
loss, acc = loaded_model.evaluate(validation_generator)
print(f"Loss: {loss:.4f}, Accuracy: {acc * 100:.2f}%")
