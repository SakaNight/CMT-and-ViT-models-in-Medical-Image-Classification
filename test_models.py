import torch
from model_vit_small import ViTClassifier
from model_cmt import create_cmt_model

def test_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vit_model = ViTClassifier(num_classes=4, attention_type='standard')
    vit_model.to(device)
    vit_input = torch.randn(1, 3, 224, 224).to(device)
    vit_output = vit_model(vit_input)
    print(f"ViT Output Shape: {vit_output.shape}")
    
    for attention_type in ['standard', 'sparse', 'local']:
        cmt_model = create_cmt_model((3, 224, 224), num_classes=4, attention_type=attention_type)
        cmt_model.to(device)
        cmt_input = torch.randn(1, 3, 224, 224).to(device)
        cmt_output = cmt_model(cmt_input)
        print(f"CMT ({attention_type}) Output Shape: {cmt_output.shape}")

if __name__ == "__main__":
    test_models()
