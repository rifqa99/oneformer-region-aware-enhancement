from PIL import Image
from src.models.oneformer_wrapper import OneFormerWrapper

model = OneFormerWrapper(device="cpu")
img = Image.open("data/sample/sample.jpg").convert("RGB")
seg = model.predict(img)
print(seg.shape)
