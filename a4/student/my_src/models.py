import torch
import torch.nn as nn
from torch.nn import functional as F

import attention

torch.manual_seed(0)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1       # embedding dropout
    resid_pdrop = 0.1      # residual dropout
    attn_pdrop = 0.1       # attention dropout
    rope = False           # whether to use RoPE
    bottleneck_dim = None  # bottleneck dimension for RoPE

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size

        for k, v in kwargs.items():
            setattr(self, k, v)  # 设置其他参数

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = attention.CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if not config.rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))  # (1, block_size, n_embd)
        
        self.drop = nn.Dropout(config.embd_pdrop)
        self.rope = config.rope

        # transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.block_size = config.block_size
        self.apply(self.__init__weights)  # 把函数当做参数apply进去，对每个module都调用一次
        print(f"number of parameters: {sum(p.numel() for p in self.parameters)}")
    
    def __init__weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_block_size(self):
        return self.block_size
    
    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward, model block size ({t}, {self.block_size}) is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)
        if self.rope:
            x_input = token_embeddings
        else:
            position_embeddings = self.pos_emb[:, :t, :]
            x_input = token_embeddings + position_embeddings # (b, t, n_embd)
        
        x = self.drop(x_input)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        
        return logits, loss
    