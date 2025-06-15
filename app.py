from flask import Flask, request, jsonify
from tensorflow import keras
import joblib
import numpy as np
import librosa
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

# تحميل الموديل الجديد والـ encoder والـ scaler
model = keras.models.load_model('./baby_cry_model_best_new20.keras')
encoder = joblib.load('./label_encoder_new20.pkl')
scaler = joblib.load('./scaler_new20.pkl')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# استخراج الخصائص من ملف الصوت وتطبيق الـ scaler
def extract_features(file_path):
    data, _ = librosa.load(file_path, sr=16000, res_type='kaiser_fast')  # تأكد من sr = 16000 زي التدريب
    mfccs = librosa.feature.mfcc(y=data, sr=16000, n_mfcc=40)
    feature = np.mean(mfccs.T, axis=0)
    feature_scaled = scaler.transform([feature])
    return feature_scaled

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = 'temp.wav'
    file.save(file_path)

    try:
        features = extract_features(file_path)  # (1, 40)
        features = np.expand_dims(features, axis=2)  # (1, 40, 1)

        predictions = model.predict(features)
        probabilities = predictions.flatten()

        categories = encoder.classes_

        class_probabilities = {category: f"{prob*100:.2f}%" for category, prob in zip(categories, probabilities)}
        predicted_label = categories[np.argmax(probabilities)]

        os.remove(file_path)

        return jsonify({'prediction': predicted_label, 'probabilities': class_probabilities})

    except Exception as e:
        os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
