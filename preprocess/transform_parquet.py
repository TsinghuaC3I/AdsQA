import os
import json
from PIL import Image
from io import BytesIO
import pyarrow as pa
import pyarrow.parquet as pq


prompt="Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."


def load_data_from_jsonl(jsonl_file: str,
                         frames_root: str):
    data_list = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            problem = obj["problem"]
            answer = obj["solution"]
            vid_path = obj["video_filename"]
            
            vid_id = os.path.splitext(os.path.basename(vid_path))[0]
            frame_dir = os.path.join(frames_root, vid_id)
            if not os.path.isdir(frame_dir):
                continue
            
            imgs = sorted(os.listdir(frame_dir))[:8]
            img_bytes_list = []
            for fn in imgs:
                img_path = os.path.join(frame_dir, fn)
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    continue
                buf = BytesIO()
                img.save(buf, format="PNG")
                img_bytes_list.append(buf.getvalue())
            

            token_prefix = "<image>" * len(img_bytes_list)
            full_problem = token_prefix + problem +prompt
            
            data_list.append({
                "images": img_bytes_list,
                "problem": full_problem,
                "answer": answer
            })
    return data_list

def main():
    jsonl_file   = "trainset.jsonl"      
    frames_root  = "adv_test"     
    out_parquet  = "adsqa.parquet"

    data_list = load_data_from_jsonl(jsonl_file, frames_root)



    table = pa.Table.from_pydict({
        "images":  [item["images"]  for item in data_list],  
        "problem": [item["problem"] for item in data_list],
        "answer":  [item["answer"]  for item in data_list],
    })


    pq.write_table(table, out_parquet)


if __name__ == "__main__":
    main()