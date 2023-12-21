from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import numpy as np
import os

app = Flask(__name__)

# Set the path to the YOLO model weights file
model_path = './model/best.pt'
model = YOLO(model_path)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_images(file_paths):
    predictions = []
    for file_path in file_paths:
        results = model(file_path)
        names_dict = results[0].names
        probs = (results[0].probs.data).tolist()
        weather_result = names_dict[np.argmax(probs)]
        predictions.append((file_path, weather_result))
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        files = request.files.getlist('file')

        # Check if no file is selected
        if not files or all(file.filename == '' for file in files):
            return render_template('index.html', error='No selected file')

        # Check file extensions
        if not all(allowed_file(file.filename) for file in files):
            return render_template('index.html', error='Invalid file extension')

        # Save and process the uploaded images
        file_paths = []
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            file_paths.append(file_path)

        predictions = process_images(file_paths)

        return render_template('result.html', predictions=predictions)

    return render_template('index.html')

@app.route('/home')
def home():
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create the 'uploads' folder if it doesn't exist
    os.makedirs('uploads', exist_ok=True)

    # Run the Flask app
    app.run(debug=True)
