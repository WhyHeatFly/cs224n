import random
import argparse  # 命令行参数解析

import dataset  # 数据集模块
import models   # 模型模块
import trainer  # 训练模块

import torch
from tqdm import tqdm  # 进度条显示
from torch.utils.tensorboard import SummaryWriter

random.seed(0)

argp = argparse.ArgumentParser()  # 创建参数解析器
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
argp.add_argument('variant', help="Choose vanilla or rope")
argp.add_argument('pretrain_corpus_path', default=None)  # 预训练语料路径
argp.add_argument('--reading_params_path', default=None)  # 读取模型参数路径
argp.add_argument('--writing_params_path', default=None)  # 写入模型参数路径
argp.add_argument('--finetune_corpus_path', default=None)  # 微调语料路径
argp.add_argument('--eval_corpus_path', default=None)  # 评估语料路径
argp.add_argument('--outputs_path', default=None)  # 输出路径
argp.add_argument('--pretrain_lr', default=6e-3, type=float)  # 预训练学习率
argp.add_argument('--finetune_lr', default=6e-4, type=float)  # 微调学习率
argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                  default='run')  # TensorBoard实验名称

args = argp.parse_args()  # 解析命令行参数

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()  # 使用当前CUDA设备

# TensorBoard trainging log
writer = SummaryWriter(log_dir='expt/%s/%s_%s_pt_lr_%f_ft_lr_%f' % (
    args.function,
    args.tb_expt_name,
    args.variant,
    args.pretrain_lr,
    args.finetune_lr))

block_size = 128
text = open(args.pretrain_corpus_path, encoding='utf-8').read()  # 读取预训练语料
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)  # 创建预训练数据集

mconf = models.GPTConfig(
    pretrain_dataset.vocab_size,
    pretrain_dataset.block_size,
    n_layer=4,  # 层数
    n_head=8,  # 注意力头数
    n_embd=256)

model = None
if args.variant == 'vanilla':
    model = models.GPT(mconf).to(device)
elif args.variant == 'rope':
    mconf.rope = True
    model = models.GPT(mconf).to(device)
else:
    raise ValueError("Unknown model variant")

print('Model on device:', next(model.parameters()).device)

if args.function == 'pretrain':
    assert args.writing_params_path is not None

    pretrain_data = open(args.pretrain_corpus_path, encoding='utf-8').read()
    pretrain_dataset = dataset.CharCorruptionDataset(pretrain_data, block_size)
    tconf = trainer.TrainerConfig(max_epochs=650, batch_size=256, learning_rate=args.pretrain_lr,
                                  lr_decay=True, warmup_tokens=512*20, final_tokens=650*len(pretrain_dataset)*block_size,
                                  num_workers=0, writer=writer)
    trainer = trainer.Trainer(model, pretrain_dataset, None, tconf)
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)
elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None

    if args.reading_params_path is not None:
        model.load_state_dict(torch.load(args.reading_params_path))
    
    pretrain_data = open(args.pretrain_corpus_path, encoding='utf-8').read()
    finetun_data = open(args.finetune_corpus_path, encoding='utf-8').read()

    pretrain_dataset = dataset.CharCorruptionDataset(pretrain_data, block_size)
    finetun_dataset = dataset.NameDataset(pretrain_dataset, finetun_data)
    
    tconf = trainer.TrainerConfig(max_epoches=75, batch_size=256, learning_rate=args.finetune_lr, 
                                  lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,
                                  num_workers=0, writer=writer)
    trainer = trainer.Trainer(model, finetun_dataset, None, tconf)
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)

