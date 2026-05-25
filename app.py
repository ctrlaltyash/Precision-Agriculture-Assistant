import os
import json
import torch
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch.nn as nn

app = Flask(__name__)

# Configuration
MODEL_PATH = 'plant_disease_model.pth'
CLASSES_PATH = 'classes.json'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Classes
try:
    with open(CLASSES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} classes.")
except Exception as e:
    print(f"Error loading classes: {e}")
    class_names = []

# Load Model
print("Loading model...")
device = torch.device("cpu") # Use CPU for inference on server
try:
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_class = class_names[predicted_idx.item()]
        return predicted_class, confidence.item()
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0

@app.route('/')
def index():
    return render_template('index.html', classes=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        label, confidence = predict_image(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({'label': label, 'confidence': confidence})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
