import torch
import torchvision.models as models

# 🔹 DeepLabV3 MobileNetV3 letöltése
print("🔹 Modell letöltése folyamatban...")

model = models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")

# 🔹 Modell mentése a fájlba
torch.save(model.state_dict(), "deeplabv3_mobilenet_v3.pth")

print("✅ Modell sikeresen letöltve és elmentve: deeplabv3_mobilenet_v3.pth")
