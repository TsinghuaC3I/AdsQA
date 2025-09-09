import os
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np

def extract_8frames_via_utils(video_path: str, out_root: str):

    base = os.path.splitext(os.path.basename(video_path))[0]    
    out_dir = os.path.join(out_root, base)
    os.makedirs(out_dir, exist_ok=True)   


    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": 1.0}
        ]
    }]

    image_inputs, video_inputs = process_vision_info(
        messages,
        min_frames=8,
        max_frames=8
    )

    frames = video_inputs[0]

 
    if hasattr(frames, "permute"):
         frames = [
            Image.fromarray(
                (frame.permute(1,2,0).cpu().numpy().clip(0,1) * 255)
                .astype(np.uint8)
            )
            for frame in frames
        ]


    for idx, img in enumerate(frames):
        fname = f"{base}_f{idx:02d}.png"
        img.save(os.path.join(out_dir, fname))

def batch_extract_8frames(video_folder: str, output_root: str):

    exts = {".mp4", ".mov", ".avi", ".mkv"}
    for fn in os.listdir(video_folder):
        if os.path.splitext(fn)[1].lower() in exts:
            path = os.path.join(video_folder, fn)
            print(f"Processing {fn} ...")
            extract_8frames_via_utils(path, output_root)
if __name__ == "__main__":
    video_folder = "data/adv"
    output_root = "dataset/adv/adv_test"
    batch_extract_8frames(video_folder, output_root)