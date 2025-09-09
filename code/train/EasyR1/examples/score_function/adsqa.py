import json
import csv
import copy
import os
import re
import socket
import sys
import hashlib
import json
import random
import numpy as np
from tqdm import tqdm
import time
import random
from openai import OpenAI
import re
from typing import Dict
# from gen_api_2 import generate_qwen
from mathruler.grader import extract_boxed_content, grade_answer


openai_api_key = "YOUR_API_KEY"
client1 = OpenAI(
    api_key=openai_api_key,
)


prompt = """
You are an advertising expert specializing in evaluating whether a respondent's answer after watching a video matches the golden answer. We will provide the video's Meta-Information, Question, Golden Answer, and the Response to be judged below.\n
###The meta-information includes the advertisement video's theme, creative points, and a brief content description, which can be regarded as ground-truth information, as follows::
{meta_info}

###Question: 
{question}

###Golden Answer: 
{golden_answer}

###Rule:
1. If the response to be judged contains ALL key information of the golden answer or expresses the same meaning using other sentences or synonyms, it is considered a match with the golden answer, and the output is 1.
2. If the response to be judged does NOT contain the key information from the golden answer, it is considered a mismatch, and the output is 0.
3. The response to be judged should NOT contain any content that is contradictory, conflicting, or unreasonable when inferred from the meta-information. If such content exist, it is considered a mismatch, and the output is 0.
4. If the response to be judged contains the MOST of key information of the golden answer and, do NOT contain any information that is contradictory, conflicting, or unreasonable when inferred from the meta-information, it is considered a partial match, and the output is 0.5.

###Response to be judged: 
{response}

###Instructions:
Follow the format below and do not give any extra outputs:
Answer: 0 (if the response does not match)
Answer: 0.5 (if the response partially match)
Answer: 1 (if the response matches)

"""

def load_jsonl_file(jsonl_file):
    # 读取整个jsonl文件到内存
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))  # 读取并解析每一行JSON数据
    return data

def find_data_by_solution(data, solution):
    # 根据solution来定位数据
    results = ""
    for item in data:
        if item.get('solution') == solution:
            return item
    return results




def generate_qwen(answer,golden):
    answer_math = re.search(r'<answer>(.*?)</answer>', answer)
    answer = answer_math.group(1).strip() if answer_math else answer.strip()
    data=load_jsonl_file("data/adv/trainset.jsonl")
    item=find_data_by_solution(data=data,solution=golden)

    prompt1=prompt.format(meta_info=item["meta_info"],question=item.get("problem"),golden_answer=golden,response=answer)
    messages1 = [
        {"role": "user", "content": prompt1},
    ]
    retries=0
    while retries < 10:
        try:
            completion1 = client1.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=messages1,
                max_tokens=80,
                temperature=0.0,
                timeout=150
            )

            break  # 成功调用后退出重试循环
        except Exception as e:
            retries += 1
            print(f"调用GPT出错，错误信息: {e}. 正在重试 {retries} 次...")
            time.sleep(5)  # 暂停5秒后重试
    # print(completion1.choices[0].message.content)
    return completion1.choices[0].message.content,answer


def format_reward(predict_str: str) -> float:
    """
    如果 predict_str 同时包含 <think>...</think> 和 <answer>...</answer> 标签，则返回 1.0，
    否则返回 0.0。
    """
    has_think = "<think>" in predict_str and "</think>" in predict_str
    has_answer = "<answer>" in predict_str and "</answer>" in predict_str
    return 1.0 if (has_think and has_answer) else 0.0


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    按照 openended_reward 的思路计算单个样例的得分：
    1. 先调用 generate_qwen(predict_str, ground_truth) 拿到 (answer, only_answer)
    2. 在 answer 中用正则提取 “Answer: <数字>”
       - 若数字为 1 → 满分 1.0，除非 only_answer 过长（>100×参考答案长度），
         此时只给 0.1
       - 若数字为 0.5 → 基础得分 0.5，同样受长度惩罚
       - 其它情况或无法匹配 → 得分 0.0
    """
    reward = 0.0
    # # 记录时间(与 openended_reward 保持一致，可按需删除)
    # _current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # symbolic verification / string-matching 入口
    answer, only_answer = generate_qwen(predict_str, ground_truth)

    pattern = re.compile(r"Answer:\s*(\d+(?:\.\d+)?)")
    match = pattern.search(answer)
    if match:
        number = match.group(1)
        # “1” ⇒ 满分；“0.5” ⇒ 半分
        if number == "1":
            reward = 0.1 if len(only_answer) > 100 * len(ground_truth) else 1.0
        elif number == "0.5":
            reward = 0.1 if len(only_answer) > 100 * len(ground_truth) else 0.5

    return reward


def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.1) -> Dict[str, float]:
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)  # handle qwen2.5vl-32b format
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward(predict_str, ground_truth)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }

if __name__ == "__main__":
    # 如果真正的 generate_qwen 不可用，就用一个最简替代实现
    # try:
    #     generate_qwen
    # except NameError:
    #     def generate_qwen(pred_str: str, ground_truth: str):
    #         # mock: 始终返回正确答案
    #         return ("Answer: 1", "1")

    # 示例预测字符串 & 参考答案
    prediction = "<think>some reasoning...</think><answer> Through entertaining content that avoids traditional ads.</answer>"
    gold = "Through entertaining content that avoids traditional ads."

    print("format_reward  :", format_reward(prediction))
    print("accuracy_reward:", accuracy_reward(prediction, gold))
    print("compute_score  :", compute_score(prediction, gold))