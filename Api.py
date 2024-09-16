import os
from flask import Flask, render_template, request, jsonify
import datetime
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')
np.set_printoptions(suppress=True)

# Carregue o modelo e os rótulos uma vez no início
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'modelo', 'keras_Model.h5')
labels_path = os.path.join(base_dir, 'modelo', 'labels.txt')

# Carregar o modelo
model = load_model(model_path, compile=False)

# Carregar os rótulos
with open(labels_path, 'r') as file:
    class_names = file.readlines()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/keras', methods=['POST'])
def uploadKeras():
    try:
        file = request.files['imagem']
        
        # Prepare a imagem para o modelo
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(file.stream).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Faça a previsão com o modelo
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # .strip() remove espaços em branco
        confidence_score = prediction[0][index]

        # Print prediction e confidence score
        print("Class:", class_name)
        print("Confidence Score:", confidence_score)

        data_atual = datetime.datetime.now()
        horario_atual = data_atual.strftime('%Y-%m-%d %H:%M:%S')

        # Mapeia o índice para os nomes das classes
        if index == 0:
            class_name = "Pneumonia"
        elif index == 1:
            class_name = "Normal"
        else:
            class_name = "Covid"

        return jsonify({
            'data': f'{horario_atual}',
            'class': f'{class_name}',
        })
    
    except Exception as e:
        # Handle exceptions and return an error message
        print("An error occurred:", e)
        return jsonify({'error': 'An internal error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)