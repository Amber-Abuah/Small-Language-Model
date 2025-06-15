import torch
import torch.nn as nn

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_heads, num_layers):
        super().__init__()

        dropout_rate = 0.2
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout_rate, dim_feedforward=d_model*4)

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len).unsqueeze(0).to(x.device)  
        x = self.embedding(x) + self.pos_embedding(positions) 
        x = self.dropout(x)
        x = x.transpose(0, 1) 
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = self.transformer(x, mask)
        x = x.transpose(0, 1)  
        x = self.dropout(x)
        return self.fc(x)
    
    def freeze_layers(self, n_layers):
        for i, layer in enumerate(self.transformer.layers):
            if i < n_layers:
                print("Freezing layer", i)
                for param in layer.parameters():
                    param.requires_grad = False
    
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)