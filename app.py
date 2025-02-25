from flask import Flask, request, send_file
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os

app = Flask(__name__)

# DeepLabV3+ AI modell betöltése
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

def segment_walls(image_path):
    """AI felismeri a falakat a képen."""
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)["out"][0]

    mask = output.argmax(0).byte().numpy()
    WALL_CLASS_ID = 15  
    wall_mask = (mask == WALL_CLASS_ID).astype(np.uint8) * 255  

    return wall_mask

@app.route('/process', methods=['POST'])
def process_image():
    image_file = request.files['image']
    color_hex = request.form['color']
    filename = "uploaded.jpg"
    image_file.save(filename)

    image = cv2.imread(filename)
    mask = segment_walls(filename)

    color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  

    alpha = 0.6
    colored_image = image.copy()
    for c in range(3):
        colored_image[:, :, c] = np.where(mask == 255,
                                          image[:, :, c] * (1 - alpha) + color_bgr[c] * alpha,
                                          image[:, :, c])

    output_filename = "output.jpg"
    cv2.imwrite(output_filename, colored_image)
    return send_file(output_filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)