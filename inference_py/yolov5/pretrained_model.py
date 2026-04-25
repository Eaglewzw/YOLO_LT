import timm
import torch
from PIL import Image
import requests
from timm.data import resolve_data_config, create_transform

# Load the model
model = timm.create_model(
    'hf_hub:jameslahm/lsnet_b',
    pretrained=True
)
model.eval()

# Load and transform image
# Example using a URL:
url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
img = Image.open(requests.get(url, stream=True).raw)

config = resolve_data_config({}, model=model)
transform = create_transform(**config)
input_tensor = transform(img).unsqueeze(0) # transform and add batch dimension

# Make prediction
with torch.no_grad():
    output = model(input_tensor)
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get top 5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)
# Assuming you have imagenet labels list 'imagenet_labels'
# for i in range(top5_prob.size(0)):
#     print(imagenet_labels[top5_catid[i]], top5_prob[i].item())
