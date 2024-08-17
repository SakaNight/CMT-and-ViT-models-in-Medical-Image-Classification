import torch
from collections import OrderedDict
from model_vit_small import ViTClassifier

checkpoint = torch.load('your_local_folder/results/ViT_NIH_sparse/Vit_NIH_sparse_trained_model.pth', map_location=torch.device('cpu'))

state_dict = OrderedDict()
for key, value in checkpoint.items():
    state_dict[key] = value

vit_model = ViTClassifier(num_classes=4, attention_type='sparse')
vit_model.load_state_dict(state_dict)
vit_model.eval()
print("Model loaded successfully.")
