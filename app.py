from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Dapatkan jalur absolut ke direktori yang berisi skrip ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Jalur ke file model yang telah dilatih
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Hama Sawi-pest-95.99.h5')

# Muat modelnya
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Kamus untuk memetakan kelas prediksi ke label
dic = {
    0: 'Hama terdeteksi',
    1: 'Tidak ada hama terdeteksi'
}

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return dic[predicted_class]

@app.route("/", methods=['POST'])
def index():
    identification_result = None
    img_filename = None
    if request.method == 'POST':
        img = request.files['my_image']
        img_filename = img.filename
        img_path = os.path.join(app.root_path, 'static', img_filename)
        try:
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            img.save(img_path)
            identification_result = predict_label(img_path)
        except Exception as e:
            print("Error:", e)  # Cetak pesan kesalahan untuk diagnosis
            return jsonify({'error': str(e)})

    return jsonify({'identificationResult': identification_result, 'imgFilename': img_filename})

if __name__ == '__main__':
    app.run(debug=True)
