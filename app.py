import os
from flask import Flask, request, render_template, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

app = Flask(__name__)

# Define constants
MODEL_PATH = "models/inceptionv3.pth"
NUM_CLASSES = 3
CLASS_NAMES = ["Armyworm", "Cutworm", "Red_Spider_Mites"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define preprocessing pipeline (same as val_test_transforms in training)
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes, device):
    """Load the trained InceptionV3 model."""
    print("Loading model...")
    try:
        model = models.inception_v3(pretrained=False, aux_logits=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise 

def preprocess_image(image):
    """Preprocess a PIL image for inference."""
    try:
        image = image.convert('RGB')
        image_tensor = PREPROCESS(image)
        image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(model, image_tensor, device, class_names):
    """Predict class label and probabilities for an image tensor with error handling."""
    try:
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = class_names[predicted_class_idx]
            confidence = probabilities[0, predicted_class_idx].item()

        return predicted_class, confidence, probabilities.cpu().numpy()[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None

# Load model at startup
model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload, location data, and return prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        # Extract latitude and longitude from the request
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')

        # Validate latitude and longitude
        if latitude is None or longitude is None:
            return jsonify({'error': 'Latitude and longitude are required'}), 400

        try:
            latitude = float(latitude)
            longitude = float(longitude)
            if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                return jsonify({'error': 'Invalid latitude or longitude values'}), 400
        except ValueError:
            return jsonify({'error': 'Latitude and longitude must be numeric'}), 400

        image = Image.open(file)
        image_tensor = preprocess_image(image)
        if image_tensor is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400

        predicted_class, confidence, probabilities = predict_image(
            model, image_tensor, DEVICE, CLASS_NAMES
        )
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Log the prediction and location (you can save this to a database or file)
        print(f"Prediction: {predicted_class}, Confidence: {confidence}, Location: ({latitude}, {longitude})")

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probabilities)},
            'latitude': latitude,
            'longitude': longitude
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Serve the HTML frontend."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)