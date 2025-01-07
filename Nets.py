import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Small2VGG Model for MNIST
class CNNMNIST(nn.Module):
    def __init__(self):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# BasicBlock for ResNet18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet Model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Bottleneck for ResNet50 and ResNet101
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



class CharLSTM(nn.Module):
    def __init__(self):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
        self.drop = nn.Dropout()
        self.out = nn.Linear(256, 80)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.drop(x)
        return self.out(x[:, -1, :])  # Take the last output
    



class Sent140LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=100, num_classes=2, seq_len=25, num_layers=2):
        super(Sent140LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size + 1, embedding_dim)  # +1 for unknown token
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.out = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embed(x)  # Embed the input
        x, _ = self.lstm(x)  # Pass through LSTM
        x = self.fc1(x[:, -1, :])  # Use the last hidden state
        x = self.dropout(x)
        x = self.out(x)  # Final output
        return x
    



class RedditTransformer(nn.Module):
    def __init__(self, vocab_size, emsize=400, nhead=8, nhid=400, 
                 nlayers=4, dropout=0.2, max_seq_length=64):
        super(RedditTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.emsize = emsize
        self.src_mask = None
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, emsize)
        self.pos_encoder = PositionalEncoding(emsize, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=emsize,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        # Decoder to predict next token
        self.decoder = nn.Linear(emsize, vocab_size)
        
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        """Generate mask for autoregressive prediction"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_padding_mask: Optional tensor indicating which elements in src are padding
                            shape [batch_size, seq_len]
        Returns:
            output: Tensor containing log probabilities for next token prediction
                   shape [batch_size, seq_len, vocab_size]
        """
        if self.src_mask is None or self.src_mask.size(0) != src.size(1):
            mask = self.generate_square_subsequent_mask(src.size(1))
            self.src_mask = mask.to(src.device)
            
        # Embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.emsize)  # [batch_size, seq_len, emsize]
        src = self.pos_encoder(src)
        
        # Transformer encoder with masking for autoregressive prediction
        output = self.transformer_encoder(
            src, 
            mask=self.src_mask,
            src_key_padding_mask=src_padding_mask
        )
        
        # Decode to get next token predictions
        output = self.decoder(output)  # [batch_size, seq_len, vocab_size]
        
        return output

    def get_attention_weights(self):
        """Returns attention weights from all layers if available"""
        weights = []
        for layer in self.transformer_encoder.layers:
            if hasattr(layer.self_attn, 'attention_weights'):
                weights.append(layer.self_attn.attention_weights)
        return weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)