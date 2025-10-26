import argparse
import time
import os
import json
import random
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import numpy as np

def read_json(jpath):
    with open(jpath, 'r') as ff:
        data = json.load(ff)

    return data


def get_qwen2vl_response(model, processor, video_inputs, video_name, question, video_kwargs):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_name,
                    "max_pixels": 360 * 420 * 4,
                    "fps": 2.0,
                },
                {"type": "text", "text": question},
            ],
        }
    ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=None,
        videos=video_inputs,
        # fps=1,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs,
                                   do_sample=True,
                                   temperature=1,
                                   max_new_tokens=1524
                                   )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()


def set_random_seed(seed):
    """
    设置PyTorch、Python和NumPy的随机数种子以确保结果可复现

    参数:
        seed (int): 随机数种子
    """
    # 设置PyTorch的随机数种子
    torch.manual_seed(seed)

    # 如果使用CUDA（GPU），还需要设置CUDA的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True  # 确保卷积操作的结果确定
        torch.backends.cudnn.benchmark = False  # 关闭自动优化，保证结果一致

    # 设置Python内置随机数种子
    random.seed(seed)

    # 设置NumPy的随机数种子
    np.random.seed(seed)

    print(f"所有随机数种子已设置为: {seed}")

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--video_dir', type=str, default='')  # dir to save videos
    args.add_argument('--file_dir', type=str, default='../adsqa_full_set')  # dir to save the question and groundtruth file
    args.add_argument('--model_dir', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")  # dir to save the checkpoints, e.g., qwen2.5-vl
    args.add_argument('--model_name', type=str, default='qwen2d5-vl-7b')  # prediction file name. This script will create a directory for each question (named with its question_id), inside which the prediction result from each model will be saved using the given prediction name.
    # args.add_argument('--run_count', type=int, default=3) # 重复实验的次数

    args = args.parse_args()

    set_random_seed(42)

    asr_reslts_set = read_json(f'./{args.file_dir}/asr_set.json')
    raw_test_data = read_json(f'./{args.file_dir}/adsqa_question_fullset.json')
    pretrained = args.model_dir

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(pretrained)

    total_nums = 0
    acc_nums = 0
    nn =0
    last_video = ""

    for ii, item in enumerate(raw_test_data):
        print(ii)
        start = time.time()
        video_name = item['video']
        question = item['question']
        question_id = item['question_id']

        asr = asr_reslts_set[video_name + '.mp4']
        if asr != '':
            asr = f"Voiceover: {asr}\n"

        if os.path.exists(os.path.join('./results', question_id, f'{args.model_name}.json')):
            continue

        try:
            video_path = os.path.join(args.video_dir, video_name + '.mp4')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            # "max_pixels": 768 * 28 * 28, # !!! 分辨率相比之前的版本 扩大了 4 倍
                            "fps": 2.0,
                        },
                    ],
                }
            ]
            if last_video != video_name:
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                last_video = video_name

            input_question = f"{asr}\n\nBy watching the video, you are required to answer a question within 30 words: Question: {question}"

            predictions = []
            for i in range(1):
                pred = get_qwen2vl_response(model, processor, video_inputs, video_name, input_question, video_kwargs)
                predictions.append(pred)

            if not os.path.exists(os.path.join('./results', question_id)):
                os.makedirs(os.path.join('./results', question_id))

            res = []
            for i in range(1):
                res.append({
                    "prediction": predictions[i],
                    "score": ""
                })

            with open(os.path.join('./results', question_id, f'{args.model_name}.json'), 'w',
                      encoding='utf-8') as ff:
                json.dump(res, ff, ensure_ascii=False, indent=4)

            print(time.time() - start)

        except Exception as ex:
            print(ex)
            print(video_name)
