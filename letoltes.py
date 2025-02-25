import torch
import torchvision.models as models

# Modell mentési fájl neve
MODEL_PATH = "deeplabv3_resnet50.pth"

# Modell letöltése PyTorch szerverről
print("🔹 Modell letöltése...")
model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)

# Modell mentése a fájlba
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Modell sikeresen letöltve és elmentve: {MODEL_PATH}")
