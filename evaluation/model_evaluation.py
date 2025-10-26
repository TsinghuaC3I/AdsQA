import argparse
import json
import csv
import copy

import math
import multiprocessing
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
import openai
import json




# First, provide your reasoning on whether the response aligns with the golden answer.
# Then, leave a blank line and provide the final result in the following format:
# Ensure that the "Answer:" field is always included.
# Note that you have to be very strict.
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

def read_json(jpath):
    with open(jpath, 'r') as ff:
        data = json.load(ff)

    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, default='readr-step240.json') # prediction file name, e.g., readr-step240.json, qwen2d5-vl-7b.json
    parser.add_argument('--test_file', type=str, default='./testset_groundtruth.json') # ground-truth file path
    parser.add_argument('--results_dir', type=str, default='./results/') # dir you save the prediction files
    parser.add_argument('--api_key', type=str, default='')  # gpt4o key
    parser.add_argument('--api_base', type=str, default='') # gpt4o url base
    args = parser.parse_args()
    print(args.test_file)
    raw_test_data = read_json(args.test_file)

    openai.api_key = args.api_key
    openai.api_base = args.api_base

    strict_acc_scores = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}
    strict_acc_counts = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}

    relax_acc_scores = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}
    relax_acc_counts = {"Type_1": 0, "Type_2": 0, "Type_3": 0, "Type_4": 0, "Type_5": 0}

    if True:

        # 定义一个最大重试次数
        max_retries = 100

        pred_nums = 0
        relaxed_acc = 0.
        strict_acc = 0.
        # 遍历每个子文件夹对应的数据项
        predict = []
        for ii, item in enumerate(raw_test_data):

            start = time.time()
            # video_name = item['video']
            question_id = item['question_id']
            meta_info = item['meta_info']
            gt_answer = item['answer'] if 'answer' in item else item['gt_answer']
            question = item['question']

            question_types = item['question_type']

            pred_path = os.path.join(args.results_dir, question_id, args.eval_name)


            if not os.path.exists(pred_path):
                continue # no found prediction file
            else:

                try:
                    pred_item = read_json(pred_path)
                except json.decoder.JSONDecodeError:
                    print(question_id)
                    continue

                pred_score = pred_item[0]['score']
                if pred_score != "":
                    gptscore = pred_score.replace('Answer: ', '').strip()
                    pred_answer = pred_item[0]['prediction']
                    if pred_answer is None:
                        # pred_answer = pred_item[0]['raw_output']
                        continue
                    pred_nums += 1
                    if '1' in gptscore:
                        strict_acc += 1
                        relaxed_acc += 1
                    elif '0.5' in gptscore:
                        strict_acc += 0
                        relaxed_acc += 0.5
                    else:
                        strict_acc += 0
                        relaxed_acc += 0

                    for typee in question_types:
                        if '1' in gptscore:
                            strict_acc_scores[typee] += 1
                            relax_acc_scores[typee] += 1
                        elif '0.5' in gptscore:
                            strict_acc_scores[typee] += 0
                            relax_acc_scores[typee] += 0.5
                        else:
                            strict_acc += 0
                            relaxed_acc += 0

                        strict_acc_counts[typee] += 1
                        relax_acc_counts[typee] += 1
                    continue # the sample has been evaluated before

            pred_answer = pred_item[0]['prediction']

            if len(pred_answer.split()) > 30:
                pred_answer = ' '.join(pred_answer.split()[0:30])

            prompt1 = prompt.format(meta_info=meta_info, question=question,
                                        golden_answer=gt_answer, response=pred_answer)
            messages1 = [
                {"role": "user", "content": prompt1},
            ]
            retries = 0
            while retries < max_retries:
                try:
                    completion1 = openai.ChatCompletion.create(
                        model="gpt-4o-2024-08-06",
                        messages=messages1,
                        max_tokens=8000,
                        temperature=0.0,
                        timeout=150
                    )

                    break  # break if success
                except Exception as e:
                    retries += 1
                    print(f"调用GPT出错，错误信息: {e}. 正在重试 {retries}/{max_retries} 次...")
                    time.sleep(5)

            gptscore = completion1.choices[0].message.content

            pred_item[0]['score'] = gptscore
            with open(pred_path, 'w') as ff:
                json.dump(pred_item, ff, indent=4, ensure_ascii=False)

            try:
                pred_nums += 1
                gptscore = gptscore.replace('Answer: ', '').strip()
                if '1' in gptscore:
                    strict_acc += 1
                    relaxed_acc += 1
                elif '0.5' in gptscore:
                    strict_acc += 0
                    relaxed_acc += 0.5
                else:
                    strict_acc += 0
                    relaxed_acc += 0

                for typee in question_types:
                    if '1' in gptscore:
                        strict_acc_scores[typee] += 1
                        relax_acc_scores[typee] += 1

                    elif '0.5' in gptscore:
                        strict_acc_scores[typee] += 0
                        relax_acc_scores[typee] += 0.5


                    strict_acc_counts[typee] += 1
                    relax_acc_counts[typee] += 1

                print(f"{question_id}: {gptscore}")

            except Exception as e:
                print(e)
                print(f"invalid return: {gptscore}")

        # save the model-based evaluation results in the prediction file.
        with open(f"{args.eval_name}", "w", encoding="utf-8") as ff:
            json.dump(predict, ff, indent=4, ensure_ascii=False)


        for typee in strict_acc_scores:
            print(f"{typee}: {strict_acc_scores[typee] / strict_acc_counts[typee]}")
        for typee in relax_acc_scores:
            print(f"{typee}: {relax_acc_scores[typee] / relax_acc_counts[typee]}")

        print(f"Total upload nums: {pred_nums}")
        print(f"Total target nums: {len(raw_test_data)}")

        print(f"Strict accuracy: {strict_acc / len(raw_test_data)}")
        print(f"Relaxed accuracy: {relaxed_acc / len(raw_test_data)}")

