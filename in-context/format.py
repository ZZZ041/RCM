import json

# 假设原始数据存储在 `data.json` 文件中
with open('trivia-test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 定义一个函数来将数据格式化
def reformat_data(original_data):
    formatted_data = []

    for item in original_data:
        # 为每个 "question" 和 "answers" 保持不变
        question_data = {
            "question": item["question"],
            "answers": item["answers"],
            "ctxs": []
        }

        # 将 "ctxs" 里的数据进行格式化
        for ctx in item["ctxs"]:
            formatted_ctx = {
                "id": ctx["id"],
                "title": ctx["title"],
                "text": ctx["text"],
                "score": ctx["score"],
                "has_answer": ctx["has_answer"]
            }
            question_data["ctxs"].append(formatted_ctx)

        formatted_data.append(question_data)

    return formatted_data

# 调用函数并获取格式化后的数据
formatted_data = reformat_data(data)

# 将格式化后的数据保存到新文件中
with open('formatted_trivia-test.json', 'w', encoding='utf-8') as outfile:
    json.dump(formatted_data, outfile, ensure_ascii=False, indent=4)

