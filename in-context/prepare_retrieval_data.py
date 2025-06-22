import json
import sys
import argparse
import numpy as np

import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from ralm.file_utils import print_args
from ralm.retrievers.retrieval_factory import add_retriever_args, get_retriever

RETRIEVAL_TYPES = [
    "dense",
    "sparse",
]


def main(args):
    # 将解析后的命令行参数 args 打印到指定的输出文件中，文件名通过替换扩展名生成
    print_args(args, output_file=args.output_file.replace(".json", ".args.txt"))

    print("Loading dataset...")
    if args.load_from == "hf":
        dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)
        dataset = "".join([x["text"] if x["text"] else " \n" for x in dataset])
    else:
        with open(args.dataset_path, "r") as f:
            dataset = f.read()

    print(f"Loading tokenizer {args.tokenizer_name} and {args.tokenizer_type}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, tokenizer_type=args.tokenizer_type)

    transformers.logging.set_verbosity_error()  # 设置Hugging Face Transformers 库的日志级别为ERROR,表示仅在发生错误时，才会输出日志信息。其他低级别的日志（如 INFO 或 WARNING）将被忽略。

    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")   # 使用 Hugging Face 的 tokenizer 对 dataset 进行编码（tokenization），并返回一个包含张量（torch.Tensor）的字典，用于进一步处理或输入模型
    dataset_num = encodings.input_ids.size(0)   # 计算 encodings.input_ids 张量的第一个维度（即句子的数量）
    dataset_len = encodings.input_ids.size(1)   # 计算 encodings.input_ids 张量的第二个维度（即句子的最大 token 长度）
    print("Dataset number:", dataset_num)
    print("Dataset length:", dataset_len)

    print(f"Creating retriever of type {args.retrieval_type}...")
    retriever = get_retriever(args.retrieval_type, args, tokenizer)     # 创建编码器

    prev_end_loc = 0
    data = []

    # tqdm用于显示进度条
    for begin_loc in tqdm(range(0, dataset_len, args.stride)):
        end_loc = min(begin_loc + args.max_length, dataset_len)
        target_begin_loc = prev_end_loc

        # d = retriever.retrieve(encodings.input_ids, target_begin_loc, end_loc, title=None)

        d = {
            "begin_location": target_begin_loc,
            "end_location": end_loc,
            "future": tokenizer.decode(encodings.input_ids[0, target_begin_loc:end_loc])    # 将分词器的ID序列解码回文本
        }

        data.append(d)  # 将每个片段的信息保存到data列表
        prev_end_loc = end_loc

        if end_loc >= dataset_len:
            break

    batch_size = 1000
    for i in range(0, len(data), batch_size):
        if i > 0:
            print(f"Finished processing {i}/{len(data)} strides")
        retriever.retrieve(encodings.input_ids, data[i:i+batch_size], k=args.num_docs)

    print(f"Finished processing {len(data)}/{len(data)} strides")
    print(f"Writing to {args.output_file}")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    with open(args.output_file, "w") as f:
        f.write(json.dumps(data, indent=4, cls=NumpyEncoder))
        f.write("\n")
        # json.dump(data, f, indent=4, cls=NumpyEncoder)

    print("Done!")


if __name__ == '__main__':

    # print(sys.argv)  # 打印所有命令行参数

    assert sys.argv[1] == "--retrieval_type"    # 断言检测
    retrieval_type = sys.argv[2]

    assert retrieval_type in RETRIEVAL_TYPES    # 断言检测

    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象，保存我们定义的所有命令行参数信息

    # Retrieval params
    parser.add_argument("--retrieval_type", required=True, choices=RETRIEVAL_TYPES)
    parser.add_argument("--num_docs", type=int, default=1)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="validation")

    # Model params
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--tokenizer_type", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=4)

    parser.add_argument("--output_file", required=True, type=str)

    add_retriever_args(parser, retrieval_type)

    args = parser.parse_args()  # 解析用户从命令行传递的参数，并将这些参数存储到一个对象（通常称为 args）中

    main(args)
