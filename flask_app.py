from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import librosa
import os
from werkzeug.utils import secure_filename
from sklearn.svm import SVC

app = Flask(__name__)

UPLOAD_FOLDER = '/home/prob1/mysite/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = joblib.load('/home/prob1/mysite/best_model.joblib')
scaler = joblib.load('/home/prob1/mysite/scaler.joblib')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        pitch = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_mean = np.mean(pitch[~np.isnan(pitch)])

        features = {}

        for i in range(20):
            features[f'mfcc_{i}'] = mfccs_mean[i]

        for i in range(7):
            features[f'contrast_{i}'] = contrast_mean[i]

        features['pitch'] = pitch_mean

        return features

    except Exception as e:
        raise ValueError(f"Ошибка обработки аудио: {str(e)}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('error.html', error="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', error="No selected file")

    if not (file and allowed_file(file.filename)):
        return render_template('error.html', error="Invalid file type")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        features = extract_features(filepath)
        selected_features = {
            'contrast_6': features['contrast_6'],
            'contrast_2': features['contrast_2'],
            'mfcc_14': features['mfcc_14'],
            'mfcc_5': features['mfcc_5'],
            'mfcc_3': features['mfcc_3']
        }
        input_data = pd.DataFrame([selected_features])
        scaled_data = scaler.transform(input_data)
        prediction = int(model.predict(scaled_data)[0])
        result = {
            'prediction_label': 'Человек' if prediction == 0 else 'Робот',
            'features': selected_features
        }
        return render_template('result.html', result=result)

    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return render_template('error.html', error=f"Processing error: {str(e)}")

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run()

