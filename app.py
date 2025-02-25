from flask import Flask, request, send_file
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import os

app = Flask(__name__)

# Modell fájl elérési útvonala
MODEL_PATH = "deeplabv3_resnet50.pth"  # 🔹 Kisebb modell a memóriaoptimalizálás érdekében

# Modell letöltése, ha nincs meg a szerveren
if not os.path.exists(MODEL_PATH):
    print("🔹 A modell nem található, letöltés folyamatban...")
    model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)  
    torch.save(model.state_dict(), MODEL_PATH)  # 🔹 Modell mentése fájlba
    print("✅ Modell letöltve és mentve!")

# Modell betöltése helyi fájlból
print("🔹 Modell betöltése...")
model = models.segmentation.deeplabv3_resnet50(weights=None)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.to(torch.device('cpu'))  # 🔹 Csak CPU-t használunk a memóriafogyasztás csökkentése érdekében
model.eval()
print("✅ Modell sikeresen betöltve!")

def segment_walls(image_path):
    """AI felismeri a falakat a képen."""
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():  # 🔹 Optimalizált memóriahasználat
        torch.cuda.empty_cache()  # 🔹 Memóriaoptimalizálás
        output = model(image_tensor)["out"][0]

    mask = output.argmax(0).byte().numpy()
    WALL_CLASS_ID = 15  # DeepLabV3+ fal osztályazonosítója
    wall_mask = (mask == WALL_CLASS_ID).astype(np.uint8) * 255  

    return wall_mask

@app.route('/process', methods=['POST'])
def process_image():
    """Feldolgozza a feltöltött képet és átszínezi a falakat."""
    if 'image' not in request.files or 'color' not in request.form:
        return "Hiba: Kép vagy szín hiányzik!", 400

    image_file = request.files['image']
    color_hex = request.form['color']
    filename = "uploaded.jpg"
    image_file.save(filename)

    # Kép beolvasása
    image = cv2.imread(filename)
    if image is None:
        return "Hiba: Nem sikerült beolvasni a képet!", 400

    mask = segment_walls(filename)

    # Szín átalakítása HEX → RGB → BGR (OpenCV miatt)
    try:
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    except ValueError:
        return "Hiba: Hibás színkód! Használj pl. '#FF5733' formátumot.", 400

    # Falak átszínezése
    alpha = 0.6  # Átlátszósági érték
    colored_image = image.copy()
    for c in range(3):
        colored_image[:, :, c] = np.where(
            mask == 255,
            image[:, :, c] * (1 - alpha) + color_bgr[c] * alpha,
            image[:, :, c]
        )

    # Mentés és küldés
    output_filename = "output.jpg"
    cv2.imwrite(output_filename, colored_image)
    return send_file(output_filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)