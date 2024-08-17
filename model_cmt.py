import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from pytorch_model_summary import summary

class SparseAttention(nn.Module):
    def __init__(self, num_heads, key_dim, sparsity_threshold=0.1):
        super(SparseAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = nn.MultiheadAttention(embed_dim=key_dim, num_heads=num_heads)
        self.sparsity_threshold = sparsity_threshold  

    def forward(self, query, key, value, mask=None):
        if len(query.size()) == 4: 
            batch_size, channels, height, width = query.size()
            query = query.view(batch_size, channels, -1).permute(2, 0, 1)
            key = key.view(batch_size, channels, -1).permute(2, 0, 1)
            value = value.view(batch_size, channels, -1).permute(2, 0, 1)
        
        attention_output, attention_weights = self.attention(query, key, value, need_weights=True)
        
        sparse_mask = (attention_weights > self.sparsity_threshold).float()
        sparse_attention_weights = attention_weights * sparse_mask
        sparse_attention_weights = sparse_attention_weights / (sparse_attention_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        sparse_attention_output = torch.matmul(sparse_attention_weights, value.transpose(0, 1)).transpose(0, 1)
        
        if len(query.size()) == 4:  
            sparse_attention_output = sparse_attention_output.permute(1, 2, 0).view(batch_size, channels, height, width)
        
        return sparse_attention_output, sparse_attention_weights

class LocalAttention(nn.Module):
    def __init__(self, num_heads, key_dim, local_window_size):
        super(LocalAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.local_window_size = local_window_size
        self.attention = nn.MultiheadAttention(embed_dim=key_dim, num_heads=num_heads)

    def forward(self, query, key, value, mask=None):
        if len(query.size()) == 4: 
            batch_size, channels, height, width = query.size()
            query = query.view(batch_size, channels, -1).permute(2, 0, 1)
            key = key.view(batch_size, channels, -1).permute(2, 0, 1)
            value = value.view(batch_size, channels, -1).permute(2, 0, 1)
        
        attention_output, attention_weights = self.attention(query, key, value, need_weights=True)

        seq_len = attention_weights.size(-1)
        local_mask = torch.zeros_like(attention_weights)
        for i in range(seq_len):
            start = max(0, i - self.local_window_size // 2)
            end = min(seq_len, i + self.local_window_size // 2 + 1)
            local_mask[:, i, start:end] = 1
        local_attention_weights = attention_weights * local_mask
        local_attention_weights = local_attention_weights / (local_attention_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        local_attention_output = torch.matmul(local_attention_weights, value.transpose(0, 1)).transpose(0, 1)
        
        if len(query.size()) == 4: 
            local_attention_output = local_attention_output.permute(1, 2, 0).view(batch_size, channels, height, width)
        
        return local_attention_output, local_attention_weights

class PatchEmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(PatchEmbeddingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class CMTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads, reduction_ratio, attention_type='standard'):
        super(CMTBlock, self).__init__()
        self.lpu = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.lpu_add = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        if attention_type == 'standard':
            self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads)
        elif attention_type == 'sparse':
            self.attention = SparseAttention(num_heads=heads, key_dim=in_channels)
        elif attention_type == 'local':
            self.attention = LocalAttention(num_heads=heads, key_dim=in_channels, local_window_size=7)

        self.attention_norm = nn.BatchNorm2d(in_channels)
        self.irffn_conv1 = nn.Conv2d(in_channels, in_channels * reduction_ratio, kernel_size=1)
        self.irffn_dwconv = nn.Conv2d(in_channels * reduction_ratio, in_channels * reduction_ratio, kernel_size=3, padding=1, groups=in_channels * reduction_ratio)
        self.irffn_conv2 = nn.Conv2d(in_channels * reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        lpu = self.lpu(x)
        lpu = self.lpu_add(lpu) + x

        batch_size, channels, height, width = lpu.size()
        lpu = lpu.view(batch_size, channels, -1).permute(2, 0, 1)

        if isinstance(self.attention, (nn.MultiheadAttention, SparseAttention)):
            attention_output, _ = self.attention(lpu, lpu, lpu)
        elif isinstance(self.attention, LocalAttention):
            attention_output, _ = self.attention(lpu, lpu, lpu)
        else:
            raise ValueError(f"Unsupported attention type: {type(self.attention)}")

        attention_output = attention_output.permute(1, 2, 0).view(batch_size, channels, height, width)
        attention_output = self.attention_norm(attention_output)

        irffn = self.irffn_conv1(attention_output)
        irffn = F.relu(self.irffn_dwconv(irffn))
        irffn = self.irffn_conv2(irffn)
        return irffn + attention_output

class CMT(nn.Module):
    def __init__(self, input_shape, num_classes, attention_type='standard'):
        super(CMT, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.stage1 = self._make_stage(32, 64, 2, attention_type)
        self.stage2 = self._make_stage(64, 128, 2, attention_type)
        self.stage3 = self._make_stage(128, 256, 4, attention_type)
        self.stage4 = self._make_stage(256, 512, 8, attention_type)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_stage(self, in_channels, out_channels, heads, attention_type):
        layers = [
            PatchEmbeddingBlock(in_channels, out_channels, stride=2),
            CMTBlock(out_channels, out_channels, heads, reduction_ratio=4, attention_type=attention_type)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_cmt_model(input_shape, num_classes, attention_type='standard'):
    model = CMT(input_shape, num_classes, attention_type=attention_type)
    return model

if __name__ == "__main__":
    input_shape = (3, 224, 224)
    model = create_cmt_model(input_shape, num_classes=1000, attention_type='sparse')
    print(model)

    batch_size = 1
    print(summary(model, torch.zeros((batch_size, *input_shape)), show_input=True))

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("model_structure", format="png", cleanup=True)