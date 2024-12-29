import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, spatial_encoding, edge_encoding):
        super(GraphormerLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.spatial_encoding = spatial_encoding
        self.edge_encoding = edge_encoding
        
        self.attn = GraphMultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index):
        
        centrality_encoding = self.get_centrality_encoding(x, edge_index)
        x = x + centrality_encoding
        
        x_norm = self.norm1(x)
        
        spatial_bias = self.get_spatial_bias(x_norm, edge_index)
        edge_bias = self.get_edge_bias(x_norm, edge_index)
        
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, spatial_bias=spatial_bias, edge_bias=edge_bias)
        x = x + attn_output
        
        x_norm = self.norm2(x)
        
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        
        return x
    
    def get_spatial_bias(self, x, edge_index):
        return torch.zeros((x.size(0), x.size(0)))
    
    def get_edge_bias(self, x, edge_index):
        return torch.zeros((x.size(0), x.size(0)))
    
    def get_centrality_encoding(self, x, edge_index):
        return torch.zeros_like(x)

class GraphMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(GraphMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key, value, spatial_bias=None, edge_bias=None):
        batch_size, num_nodes, embed_dim = query.size()
        
        Q = self.query_proj(query).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if spatial_bias is not None:
            attn_scores += spatial_bias.unsqueeze(1)
        if edge_bias is not None:
            attn_scores += edge_bias.unsqueeze(1)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_nodes, embed_dim)
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class Graphormer(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads):
        super(Graphormer, self).__init__()
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, num_heads, spatial_encoding=True, edge_encoding=True)
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.readout(x.mean(dim=0))

model = Graphormer(num_layers=4, hidden_dim=128, num_heads=4)
x = torch.rand((1, 10, 128))
edge_index = torch.tensor([[0, 1], [1, 2]])
output = model(x, edge_index)
print(output)