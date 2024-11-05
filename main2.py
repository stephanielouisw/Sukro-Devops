import os
import numpy as np
from time import sleep
from flask import Flask, render_template, flash, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_session import Session
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# Constants for model paths
MODELS_ARCHITECTURE = 'models/model_corn_disease_1.json'
MODELS_WEIGHT = 'models/model_corn_disease_weight.h5'

def create_ann_model():
    """
    Create and return a new ANN model for corn disease classification
    """
    model = Sequential([
        # Convolutional layers
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten layer to transition to dense layers
        Flatten(),
        
        # Dense layers (ANN part)
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')  # 4 classes for corn diseases
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Load the model architecture and weights
try:
    # Try to load existing model first
    with open(MODELS_ARCHITECTURE, 'r') as json_file:
        loaded_model_json = json_file.read()
    models = model_from_json(loaded_model_json)
    models.load_weights(MODELS_WEIGHT)
    print('@@ Existing model loaded successfully')
except Exception as e:
    print(f"Error loading existing model: {e}")
    print('@@ Creating new ANN model')
    models = create_ann_model()

def preprocess_image(img_path):
    """
    Preprocess image for model prediction
    """
    test_image = load_img(img_path, target_size=(224, 224))
    test_image = img_to_array(test_image)
    test_image = test_image / 255.0  # Normalize
    test_image = np.expand_dims(test_image, axis=0)
    return test_image

def predict(img_path, models):
    """
    Predict corn disease using the ANN model
    """
    test_image = preprocess_image(img_path)
    print('@@ Got Image for prediction')

    # Get raw predictions
    result = models.predict(test_image)
    print('@@ Raw prediction probabilities:', result[0])
    
    # Get class with highest probability
    predicted_class = np.argmax(result[0])
    
    # Get confidence score
    confidence = result[0][predicted_class]
    print(f'@@ Predicted class: {predicted_class} with confidence: {confidence:.2f}')
    
    return predicted_class, confidence

app = Flask(__name__, template_folder='web')
app.secret_key = 'algo908%jejeneverdiesiapajejesayagatau'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

KONDISI = ['BLIGHT', 'COMMON RUST', 'GRAY LEAF SPOT', 'HEALTHY']
TIPS1 = [
    'Strategi manajemen preventif dapat mengurangi kerugian ekonomi dari NCLB. Manajemen pencegahan sangat penting untuk bidang yang berisiko tinggi untuk perkembangan penyakit. Pilihan pengelolaan penyakit dalam musim, seperti fungisida, juga tersedia.',
    'Penyakit ini menyukai suhu dingin (60 - 76 derajat F), embun yang lebat, sekitar enam jam kebasahan daun, dan kelembapan relatif lebih dari 95 persen. Suhu di atas 80 derajat F menekan perkembangan dan penyebaran penyakit.',
    'Patogennya adalah jamur yang disebut Cercospora zeae-maydis. Satu-satunya inang yang diketahui adalah jagung dan melewati musim dingin di puing-puing di permukaan tanah.',
    'Waaahh.. Selamat tanaman jagung Anda SEHAT.'
]
TIPS2 = [
    'Resistensi utama (vertikal) hibrida jagung berasal dari gen spesifik ras Ht1, Ht2, Ht3, dan HtN, dengan gen Ht1 yang paling umum. Tanaman dengan gen Ht1 Ht2, atau Ht3 memiliki lesi klorotik yang lebih kecil dan sporulasi yang berkurang.[2] Gen HtN menunda gejala sampai setelah serbuk sari ditumpahkan. Secara individual, masing-masing gen Ht memiliki efektivitas terbatas karena ada ras E',
    'Praktik manajemen terbaik adalah dengan menggunakan jagung hibrida tahan. Fungisida juga bisa bermanfaat, terutama jika diterapkan lebih awal ketika beberapa pustula muncul di daun.',
    'Taktik pengelolaan penyakit termasuk menggunakan jagung hibrida yang tahan, pengolahan tanah konvensional jika sesuai, dan rotasi tanaman. Fungisida daun bisa efektif jika dijamin secara ekonomi. Biasanya mereka hanya menguntungkan pada inbrida rentan atau hibrida rentan di bawah kombinasi kondisi berisiko tinggi dengan potensi hasil tinggi, kondisi lembab berkepanjangan, dan bukti perkembangan penyakit.',
    'Tetap terus rawat tanaman Anda agar selalu terhindar dari penyakit yaa! Kenali gejala penyakit yang dapat menyerang tanaman jagung Anda juga, agar ketika tanaman jagung Anda terkena penyakit Anda bisa mengatasinya dengan cepat. Pantau juga tanaman Anda dari serangan hama agar tanaman Anda tetap sehat sampai masa panen tiba. Good Luck :)'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    sleep(2)
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predicted_class, confidence = predict(filepath, models)
            session['classes'] = str(predicted_class)
            session['confidence'] = str(confidence)
            session['filepath'] = filepath
            return redirect(url_for('hasil'))

@app.route('/dokumentasi')
def dokumentasi():
    return render_template('dokumentasi.html')

@app.route('/hasil')
def hasil():
    classes = int(session.get('classes'))
    confidence = float(session.get('confidence'))
    filepath = session.get('filepath')
    return render_template('hasil.html',
                         penyakit=KONDISI[classes],
                         confidence=f"{confidence*100:.2f}%",
                         imagepath=filepath,
                         tips1=TIPS1[classes],
                         tips2=TIPS2[classes])

@app.route('/upload')
def proses():
    return render_template('proses.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)