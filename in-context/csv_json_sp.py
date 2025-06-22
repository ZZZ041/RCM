import csv
import re
import json
import os


def process_prompt(prompt):
    """
    Extract text from line 16 to third-to-last line (inclusive of empty lines)
    """
# 找到"Answer: Pumpernickel"的位置
    last_pos = prompt.rfind("Answer: Pumpernickel")
    if last_pos != -1:
        # 跳过"Answer: Pumpernickel"字符串(长度为20)，提取后面的文本
        text_after_marker = prompt[last_pos + 20:].strip()
        # 将文本分割成行
        lines = text_after_marker.split('\n')        
        # 如果行数大于2，去掉最后两行
        if len(lines) > 2:
            extracted_lines = lines[:-2]
            return '\n'.join(extracted_lines).strip()
    return ''


def process_answers(answer_str):
    """
    Process the answer string to remove unwanted characters like brackets and quotes.
    For example, converts `['David Seville']` to `David Seville`.
    """
    # 去除方括号和引号
    answer = re.sub(r"[\'\[\]]", "", answer_str).strip()
    return answer


def convert_csv_to_json(csv_path, json_path):
    data = []

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        return

    print(f"开始处理文件 {csv_path}")

    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            print(f"成功打开文件")

            # 尝试读取第一行来验证CSV格式
            first_line = csvfile.readline()
            print(f"第一行内容: {first_line}")

            # 重置文件指针
            csvfile.seek(0)

            reader = csv.reader(csvfile)
            header = next(reader)  # 跳过标题行
            print(f"CSV标题: {header}")

            row_count = 0
            valid_row_count = 0

            for row in reader:
                row_count += 1
                if len(row) >= 3:
                    valid_row_count += 1

                    question = row[0]
                    print(f"处理问题 {row_count}: {question[:30]}...")  # 只打印问题的前30个字符

                    # 使用 process_answers 去掉多余的字符
                    answers = [process_answers(ans) for ans in row[1].split(',')]  # 处理答案
                    prompt = process_prompt(row[2])

                    # 检查处理后的内容是否为空
                    if not prompt:
                        print(f"警告：问题 {row_count} 的提示文本处理后为空")
                        # print(f"原始提示文本行数: {len(row[2].split('\n'))}")

                    entry = {
                        "question": question,
                        "answers": answers,
                        "ctxs": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                    data.append(entry)
                else:
                    print(f"警告：第 {row_count} 行数据列数不足3列: {row}")

            print(f"CSV文件共 {row_count} 行，有效数据 {valid_row_count} 行，处理后得到 {len(data)} 条记录")

        # 检查是否有数据要写入
        if not data:
            print("警告：没有找到有效数据，JSON文件将为空")

        # 写入JSON文件，让json.dumps来处理字符串的引号
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"成功写入 {len(data)} 条记录到 {json_path}")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    csv_path = 'tqa_5shot_top_5_passage_turbo_compression.csv'  # 替换为你的CSV文件路径
    json_path = 'tqa_5shot_top_5_passage_turbo_compression.json'  # 替换为目标JSON文件路径
    convert_csv_to_json(csv_path, json_path)