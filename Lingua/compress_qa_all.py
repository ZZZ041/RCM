from llmlingua import PromptCompressor
import json
from tqdm import tqdm
import torch

# 在模型初始化之前检查并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1000  # 批处理大小，根据内存情况调整

RETRIEVAL_FILE = "/home/zhangzhenzhi/Project/in-context/formatted_trivia-test.json"
OUTPUT_FILE = "formatted_trivia-test-top5-0.5big.json"


# 第一步：读取 JSON 文件
try:
    with open(RETRIEVAL_FILE, "r") as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading JSON file: {e}")
    exit(1)

# 确保 data 是一个列表
if isinstance(data, list):
    processed_data = []

    # 使用 LLMLingua-2-big 模型
    try:
        llm_lingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
        )
    except Exception as e:
        print(f"Error initializing PromptCompressor: {e}")
        exit(1)

    # 准备分批处理
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]

        # 遍历每个batch内的对象，提取 retrieved_docs，使用 tqdm 显示进度条
        for entry in tqdm(batch, desc=f"Processing batch {i // BATCH_SIZE + 1}"):
            ctxs = entry.get("ctxs", [])

            first_doc = ctxs[0]

            # 如果 retrieved_docs 为空列表
            if not ctxs:
                processed_docs = []
            else:
                # 拼接前X个文档的文本内容
                combined_text = [doc['text'] for doc in ctxs[:5]]

                processed_docs = []

                compressed_prompt = llm_lingua.compress_prompt(combined_text, rate=0.5, force_tokens=['\n', '?'])

                # 保存处理后的结果
                processed_docs.append({
                    "id": first_doc['id'],
                    "title": first_doc['title'],
                    "text": compressed_prompt.get('compressed_prompt', ""),
                    "score": first_doc['score'],
                    "has_answer": first_doc['has_answer'],
                })

            # 保存每个 entry 的处理结果
            processed_entry = {
                "question": entry.get("question"),
                "answers": entry.get("answers"),
                "ctxs": processed_docs  # 更新为处理后的文档
            }

            processed_data.append(processed_entry)

    # 所有批次处理完成后，写入文件
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(processed_data, f, indent=4)
        print(f"Processed documents written to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving processed data: {e}")
else:
    print("Error: The data is not a list.")

