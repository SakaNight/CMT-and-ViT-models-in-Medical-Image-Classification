import torch
from torchviz import make_dot
from thop import profile
from model_vit_small import ViTClassifier
from model_cmt import create_cmt_model

def print_model_summary(model, input_size):
    model.eval()
    input_tensor = torch.randn(input_size).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model layout:\n{model}")

def count_flops_torch(model, input_size=(1, 3, 224, 224)):
    inputs = torch.randn(input_size)
    flops, params = profile(model, inputs=(inputs,))
    return flops, params

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate_models():
    print("Validating ViT Model:")
    vit_model = ViTClassifier(num_classes=1000)
    vit_flops, vit_params = count_flops_torch(vit_model)
    vit_param_count = count_parameters(vit_model)
    print(f"ViT FLOPs: {vit_flops/1e9:.2f} GFLOPs")
    print(f"ViT Parameters: {vit_param_count/1e6:.2f}M")
    
    print("\nValidating CMT Model:")
    input_shape = (1, 3, 224, 224)
    cmt_model = create_cmt_model(input_shape[1:], num_classes=1000, attention_type='standard')
    cmt_flops, cmt_params = count_flops_torch(cmt_model)
    cmt_param_count = count_parameters(cmt_model)
    print(f"CMT FLOPs: {cmt_flops/1e9:.2f} GFLOPs")
    print(f"CMT Parameters: {cmt_param_count/1e6:.2f}M")

if __name__ == "__main__":
    validate_models()
