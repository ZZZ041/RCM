from llmlingua import PromptCompressor
import json
from tqdm import tqdm
import torch

'''
先压缩再拼接，压缩模型为big，保存文件为RF_lingua2big_xxx
大写RF
'''

# 在模型初始化之前检查并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1000  # 批处理大小，根据内存情况调整

RETRIEVAL_FILE = "/home/zhangzhenzhi/Project/in-context/$RETRIEVAL_FILE_S"
OUTPUT_FILE = "RF_lingua2big_5_0.34"

# 第一步：读取 JSON 文件
with open(RETRIEVAL_FILE, "r") as f:
    data = json.load(f)

# 确保 data 是一个列表
if isinstance(data, list):
    processed_data = []

    ## use LLMLingua-2-big model
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,  # Whether to use llmlingua-2
    )

    # 准备分批处理
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]

        # 遍历每个batch内的对象，提取 retrieved_docs，使用 tqdm 显示进度条
        for entry in tqdm(batch, desc=f"Processing batch {i // BATCH_SIZE + 1}"):
            retrieved_docs = entry.get("retrieved_docs", [])

            # 如果 retrieved_docs 为空列表
            if not retrieved_docs:
                processed_docs = []
            else:
                # 取前X个文档
                top_docs = retrieved_docs[:5]

                # 分别压缩每个文档的文本内容
                compressed_texts = []
                for doc in top_docs:
                    text = doc['text']
                    # 对单个文档进行压缩
                    compressed_prompt = llm_lingua.compress_prompt(text, rate=0.34, force_tokens=['\n', '?'])
                    compressed_text = compressed_prompt.get('compressed_prompt', "")
                    compressed_texts.append(compressed_text)

                # 将所有压缩后的文本拼接在一起
                combined_compressed_text = "\n\n".join(compressed_texts)

                # 保存处理后的结果
                processed_docs = [{
                    "text": combined_compressed_text
                }]

            # 保存每个 entry 的处理结果
            processed_entry = {
                "begin_location": entry.get("begin_location"),
                "end_location": entry.get("end_location"),
                "future": entry.get("future"),
                "query": entry.get("query"),
                "retrieved_docs": processed_docs  # 更新为处理后的文档
            }

            processed_data.append(processed_entry)

    # 所有批次处理完成后，写入文件
    with open(OUTPUT_FILE, "w") as f:
        json.dump(processed_data, f, indent=4)

    print(f"Processed documents written to {OUTPUT_FILE}")
else:
    print("Error: The data is not a list.")
