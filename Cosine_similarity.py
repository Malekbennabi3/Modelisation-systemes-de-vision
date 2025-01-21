# ctranspath feature extractor code here

import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from timm.models.layers.helpers import to_2tuple
import timm
import torch.nn as nn


class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def ctranspath():
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model


class CTransPath(nn.Module):

    def __init__(self):
        super(CTransPath, self).__init__()

        self.transforms = transforms.Compose(
            [
                transforms.Resize(224),
            ]
        )

        self.model = ctranspath()
        self.model.head = nn.Identity()
        td = torch.load('ctranspath.pth')
        self.model.load_state_dict(td['model'], strict=False)

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)


def ctranspath_baseline():
    """Constructs a ctranspath model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CTransPath()
    return model


#_____________________________________________________________________________________________________________________

import torch
from PIL import Image
from torchvision import transforms

# Assurez-vous que le modèle CTransPath est bien défini comme dans votre code précédent
# Ou importez-le si c'est dans un fichier à part

# Chargez le modèle pré-entrainé
model = ctranspath_baseline()

# Mettez le modèle en mode évaluation (évitant la mise à jour des poids pendant l'inférence)
model.eval()

# Charger et pré-traiter une image
img_path = "D:\\Images_WSI\\Histo\\TEST\\MSIH\\TCGA-A6-6653-01Z-00-DX1.e130666d-2681-4382-9e7a-4a4d27cb77a4_(85268,12490).jpg"  # Remplacez par le chemin de votre image
img = Image.open(img_path).convert("RGB")

# Transformation : redimensionner l'image à 224x224 et la convertir en tensor
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_tensor = transform(img).unsqueeze(0)  # Ajouter la dimension de batch (B, C, H, W)

# Passer l'image dans le modèle pour obtenir les caractéristiques
with torch.no_grad():  # Pas de calcul des gradients pendant l'inférence
    features = model(img_tensor)

# Afficher les dimensions des caractéristiques extraites
print(f"Dimensions des caractéristiques extraites : {features.shape}")


#_______________________________________________________________________________________________________________________

import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# Load the two images
img1_path = "D:\\Images_WSI\\Histo\\TEST\\MSIH\\TCGA-A6-6653-01Z-00-DX1.e130666d-2681-4382-9e7a-4a4d27cb77a4_(85268,12490).jpg"
img2_path = "D:\\Images_WSI\\Histo\\TRAIN\\nonMSIH\\TCGA-CK-5915-01Z-00-DX1.04650539-005e-4221-90d6-49706b1d7244_(99594,38926).jpg"

# Apply the necessary transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

img1 = Image.open(img1_path).convert("RGB")
img2 = Image.open(img2_path).convert("RGB")

img1_tensor = transform(img1).unsqueeze(0)  # Add batch dimension
img2_tensor = transform(img2).unsqueeze(0)

# Recuperer les features a partir des images
with torch.no_grad():
    features1 = model(img1_tensor)
    features2 = model(img2_tensor)

# Reshape the features to ensure they are 2D arrays for cosine similarity
features1 = features1.squeeze().unsqueeze(0)
features2 = features2.squeeze().unsqueeze(0)

# Compute the cosine similarity between the features of the two images
similarity = cosine_similarity(features1.numpy(), features2.numpy())

print(f"Cosine similarity between the two images: {similarity[0][0]}")


print("Mean of features1:", features1.mean().item())
print("Std of features1:", features1.std().item())
print("Mean of features2:", features2.mean().item())
print("Std of features2:", features2.std().item())
