import torch
import torch.nn as nn
import math
from transformers import ViTModel, ViTConfig
from transformers.models.vit.modeling_vit import ViTAttention
from pytorch_model_summary import summary
from torchviz import make_dot

class SparseAttention(nn.Module):
    def __init__(self, attention, sparsity_threshold=0.1):
        super(SparseAttention, self).__init__()
        self.attention = attention
        self.sparsity_threshold = sparsity_threshold

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        attention_output = self.attention(hidden_states, head_mask=head_mask, output_attentions=True)
  
        if len(attention_output) > 1 and attention_output[1] is not None:
            attn_weights = attention_output[1]
            sparse_mask = (attn_weights > self.sparsity_threshold).float()
            sparse_attn_weights = attn_weights * sparse_mask
            sparse_attn_weights = sparse_attn_weights / (sparse_attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
            
            return (attention_output[0], sparse_attn_weights) if output_attentions else (attention_output[0],)
        else:
            return attention_output

import torch
import torch.nn as nn

class LocalAttention(nn.Module):
    def __init__(self, attention, local_range=7):
        super(LocalAttention, self).__init__()
        self.attention = attention
        self.local_range = local_range

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        attention_output = self.attention(hidden_states, head_mask=head_mask, output_attentions=True)
        
        if isinstance(attention_output, tuple) and len(attention_output) > 1:
            context_layer, attn_weights = attention_output[:2]
        else:
            context_layer = attention_output
            attn_weights = None
        
        if attn_weights is not None:
            seq_len = attn_weights.size(-1)
            local_mask = torch.zeros_like(attn_weights)
            for i in range(seq_len):
                start = max(0, i - self.local_range)
                end = min(seq_len, i + self.local_range + 1)
                local_mask[:, :, i, start:end] = 1
            local_attn_weights = attn_weights * local_mask
            local_attn_weights = local_attn_weights / (local_attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
            
            context_layer = torch.matmul(local_attn_weights, hidden_states)
        else:
            local_attn_weights = None

        return (context_layer, local_attn_weights) if output_attentions else (context_layer,)

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=1000, attention_type='standard'):
        super(ViTClassifier, self).__init__()
        # by defult, ViT-Small model is used, the settings are also defult
        # config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        # use ViT-Tiny model instead of ViT-Small
        config = ViTConfig.from_pretrained('WinKawaks/vit-tiny-patch16-224')
        self.vit = ViTModel(config)

        for param in self.vit.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

        for layer in self.vit.encoder.layer:
            if attention_type == 'sparse':
                layer.attention.attention = SparseAttention(layer.attention.attention)
            elif attention_type == 'local':
                for layer in self.vit.encoder.layer:
                    layer.attention.attention = LocalAttention(layer.attention.attention)

        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(x)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

def main():
    model = ViTClassifier(num_classes=4, attention_type='sparse')
    print(model)

    batch_size = 1
    input_shape = (3, 224, 224) 
    print(summary(model, torch.zeros((batch_size, *input_shape)), show_input=True))

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("vit_model_structure", format="png", cleanup=True)

if __name__ == "__main__":
    main()