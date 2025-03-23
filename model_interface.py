# Import packages
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import torch.nn as nn
from torchvision.utils import save_image

def to_np(tensor):
    return tensor.permute( 1, 2, 0).detach().cpu().numpy()


# Transformations for the input images
img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])
gt_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            #transforms.Normalize([0.5],[0.5])
        ])

# Functions to load images
def get_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img_transform(img)
    return img

def get_image_nonorm(img_path):
    img = Image.open(img_path).convert('RGB')
    img = gt_transform(img)
    return img

def get_gt_tensor(img_path):
    img = Image.open(img_path).convert('L')
    img = gt_transform(img)

    return img
                
def load_model(model_checkpoint_path):
    state_dict = torch.load(model_checkpoint_path)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'module' not in k.split('.')[0]:
            k = 'module.' + k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k] = v
    #     print(k)
    model = nn.DataParallel(model)
    model.load_state_dict(new_state_dict)
    return model
    
# Predict saliency
def predict(image):

    with torch.no_grad():
        fin_pred , temp_pred = model(image) #Image saliency and temporal saliency predictions
        temp_pred = temp_pred.squeeze(0)

    return fin_pred , temp_pred 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to the model checkpoints
model_checkpoint_path = "./Tempsal/src/checkpoints/multilevel_tempsal.pt"
time_slices = 5
train_model = 0



from pathlib import Path
import sys

# Add the target folder to sys.path
sys.path.insert(0, str(Path(__file__).parent / "Tempsal/src"))


from model import PNASBoostedModelMultiLevel

# revert the changed path
sys.path.insert(0, str(Path(__file__).parent))
print("############")
print(Path.cwd())

model = PNASBoostedModelMultiLevel(device, model_checkpoint_path, model_checkpoint_path, time_slices, train_model=train_model )    
    
# Load model    
#model = load_model(model_checkpoint_path)
model = model.to(device)
model.eval()


image_path = "C:/Users/jerem/OneDrive/Pictures/20230115_115255.jpg"

# Read image
image = get_image(image_path)
# Predict saliency
fin_pred , temp_pred  = predict(image.unsqueeze(0))
temp_pred = temp_pred.unsqueeze(1)

# Display the image and prediction
#plt.axis("off")
img = to_np(fin_pred)

# Display the original image and the saliency map
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(to_np(image))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img, cmap='hot')
plt.axis('off')

# save the orinal image
save_image(image, "original_image.jpg")

plt.show()
