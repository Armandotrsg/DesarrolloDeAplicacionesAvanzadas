import torch
#model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
from googlenet_pytorch import GoogLeNet 
from PIL import Image
from torchvision import transforms
import os
import wget

model = GoogLeNet.from_pretrained('googlenet')
model.eval()

filename = "botella.jpeg"

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)

# Download ImageNet labels
#Check if the file exists
if not os.path.exists('imagenet_classes.txt'):
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    filename = wget.download(url)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


if "paper" in str(categories[top5_catid[0]]) or "tissue" in str(categories[top5_catid[0]]):
    tipo = "paper"
elif "beer" in str(categories[top5_catid[0]]) or "glass" in str(categories[top5_catid[0]]):
    tipo = "glass"
elif "plastic" in str(categories[top5_catid[0]]) or "bottle" in str(categories[top5_catid[0]]):
    tipo = "plastic"
elif " can" in str(categories[top5_catid[0]])  or "can " in str(categories[top5_catid[0]]):
    tipo = "metal"
else:
    tipo = "trash"

# open image with title
import matplotlib.pyplot as plt
plt.imshow(input_image)
plt.title(tipo+" "+str(int(top5_prob[0].item()*100))+"%")
plt.show()
