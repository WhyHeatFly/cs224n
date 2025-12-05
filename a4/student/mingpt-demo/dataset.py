# jupyter的魔法命令，将下面的代码保存到dataset.py 方便调用

import math
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size} characters, {vocab_size} unique.")

        self.stoi = {ch : i for i, ch in enumerate(chars)}
        self.itos = {i : ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size # 为啥 - block_size? 因为最后 block_size 个字符不能作为输入
    
    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.black_size + 1]  # 多一个字符作为target
        # encode
        dix = [self.stoi[s] for s in chunk]
        
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
