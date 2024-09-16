from flask import Flask, request, render_template, redirect, url_for
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# Carregar o modelo
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)  # Ajuste para o número de classes

# Carregar os pesos do modelo
model_path = 'model/lung_disease_model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load(model_path, map_location=device))
model_ft.eval()

# Definir o modelo globalmente
model = model_ft


# Definir as transformações de imagem
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['NORMAL', 'CANCER']

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')  # Converte a imagem para RGB
    img = data_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        prediction = class_names[preds[0]]
        print(f'Predição: {prediction}, Saídas do modelo: {outputs}')
        return prediction


@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template('index.html', prediction=prediction, image_path=file.filename)
    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
