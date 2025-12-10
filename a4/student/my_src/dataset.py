import torch
from torch.utils.data import Dataset
import random


# 预训练数据集类
class CharCorruptionDataset(Dataset):

    def __init__(self, data, block_size):
        self.MASK_CHAR = "\u2047"  # 掩码字符，双问号
        self.PAD_CHAR = "\u25A1"  # 填充字符，空白方块

        chars = list(sorted(list(set(data)))) # 获取数据中的唯一字符并排序
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size} characters, {vocab_size} unique.")

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n') # 按行分割数据,是一个字符串列表

        def __len__(self):
            return len(self.data)
        
        # 获得预训练数据集被掩码处理后的样本对
        def __getitem__(self, idx):

            document = self.data[idx]

            max_trunc_length = int(self.block_size * 7 / 8)
            trunc_length = random.randint(4, max_trunc_length)
            trunc_data = document[:trunc_length]
            # 划分前缀、被掩码内容和后缀
            masked_length = max(1, int(random.gauss(trunc_length / 4, trunc_length / 8)))
            masked_start = random.randint(1, trunc_length - masked_length - 1)
            prefix = trunc_data[:masked_start]
            masked_content = trunc_data[masked_start: masked_start + masked_length]
            suffix = trunc_data[masked_start + masked_length:]
            # 创建掩码字符串
            masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content
            masked_string += self.PAD_CHAR * (self.block_size + 1 - len(masked_string))

            input_str = masked_string[:-1]
            output_str = masked_string[1:]
            x = torch.tensor([self.stoi[c] for c in input_str], dtype=torch.long)
            y = torch.tensor([self.stoi[c] for c in output_str], dtype=torch.long)
            return x, y


# 微调数据集类
class NaneDataset(Dataset):

    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = "\u2047"
        self.PAD_CHAR = "\u25A1"
        self.itos = pretraining_dataset.itos
        self.stoi = pretraining_dataset.stoi
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))
    
    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR * (self.block_size - len(x))
        y = self.PAD_CHAR * (len(inp)) + x[len(inp):]

        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y