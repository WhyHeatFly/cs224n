import argparse 

# 创建解析器对象
parser = argparse.ArgumentParser(description="A tiny training sript demo")

parser.add_argument("mode", type=str, help="train or test")  # 为什么是位置参数？ 因为没有加--前缀
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")  # 可选参数
parser.add_argument("--epochs", type=int, default=32, help="number of epochs") # 为什么是可选参数？ 因为加了--前缀

args = parser.parse_args() # 解析命令行参数， 返回一个包含参数的命名空间对象

print(f"mode: {args.mode}")
print(f"learning rate: {args.lr}")
print(f"batch size: {args.epochs}")

if args.mode == "train":
    print(f"开始训练 {args.epochs}轮， 学习率 {args.lr}")
     