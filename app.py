from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

app = Flask(__name__)

# Load trained model and class indices
model = tf.keras.models.load_model('conv2d.h5')
with open('class_indices.json') as f:
    class_indices = json.load(f)

# Reverse the class_indices to get class names from prediction index
idx_to_class = {int(v): k for v, k in class_indices.items()}

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Image Preprocessing
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.array(img, dtype='float32') / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"Received contact form: {name}, {email}")
        return redirect(url_for('contact', submitted='true'))

    return render_template('contact.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        flash("No file selected. Please choose an image before clicking Predict.")
        return redirect(url_for('home'))

    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Predict
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)
        predicted_class = idx_to_class[np.argmax(prediction)]

        return render_template('result.html', image_path=image_path, prediction=predicted_class)

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
