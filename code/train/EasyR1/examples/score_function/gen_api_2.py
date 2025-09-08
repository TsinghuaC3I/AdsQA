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

data=load_jsonl_file("data/trainset_5.jsonl")


def generate_qwen(answer,golden):
    answer_math = re.search(r'<answer>(.*?)</answer>', answer)
    answer = answer_math.group(1).strip() if answer_math else answer.strip()

    item=find_data_by_solution(data=data,solution=golden)

    prompt1=prompt.format(meta_info=item["meta_info"],question=item.get("problem"),golden_answer=golden,response=answer)
    messages1 = [
        {"role": "user", "content": prompt1},
    ]
    completion1 = client1.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=messages1,
    max_tokens=128,
    temperature=0.0,
    timeout=150
    )
    # print(completion1.choices[0].message.content)
    return completion1.choices[0].message.content,answer
def test(answer,golden):
    answer_math = re.search(r'<answer>(.*?)</answer>', answer)
    answer = answer_math.group(1).strip() if answer_math else answer.strip()

    # item=find_data_by_solution(data=data,solution=golden)

    prompt1=prompt.format(meta_info=answer,question=answer,golden_answer=answer,response=answer)
    messages1 = [
        {"role": "user", "content": prompt1},
    ]
    completion1 = client1.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=messages1,
    max_tokens=128,
    temperature=0.0,
    timeout=150
    )
    # print(completion1.choices[0].message.content)
    return completion1.choices[0].message.content,answer

if __name__ == "__main__":
    print(test("The use of smoke and red lighting during the interactions of El Covid and The Rona creates a visually arresting and subtly threatening atmosphere","The use of smoke and red lighting during the interactions of El Covid and The Rona creates a visually arresting and subtly threatening atmosphere, highlighting the risk of close contact."))