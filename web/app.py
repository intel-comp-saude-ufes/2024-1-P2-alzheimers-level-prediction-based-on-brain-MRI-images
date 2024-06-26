import sys
import os
# Added parent directory to system path to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from flask import Flask, request, render_template
from torchvision import transforms
import torch
from PIL import Image
from io import BytesIO

from advanced_cnn import AdvancedCNN

app = Flask(__name__)

# Defining transformations for image preprocessing (same as in main.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Checking if the GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the trained model
model = AdvancedCNN()
model.load_state_dict(torch.load('../model_fold_1.pth', map_location=torch.device(device)))
model.eval() # Sets the model in evaluation mode

# Function to predict image class
def predict_image(image_bytes, model):
    image = Image.open(image_bytes).convert('L') # Opens the image and converts it to grayscale
    image = transform(image).unsqueeze(0) # Apply transformations and add extra batch dimension
    outputs = model(image) # Pass the image through the model
    _, predicted = torch.max(outputs, 1) # Get the predicted class
    return predicted.item() # Returns the class as an integer value 

# Route to home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_bytes = BytesIO(file.read())
            prediction = predict_image(image_bytes, model)
            class_names = ['non-demented', 'very-mild-demented', 'mild-demented', 'moderate-demented']
            result = class_names[prediction]
            return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)