import os
import requests
from urllib.parse import urlparse
import json
import argparse


def read_json(path):
    with open(path) as f:
        return json.load(f)


def download_video(url, save_path=None):

    try:
        # 获取文件名
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        if not filename:
            filename = "downloaded_video.mp4"

        # 设置保存路径
        if save_path:
            if os.path.isdir(save_path):
                filepath = os.path.join(save_path, filename)
            else:
                filepath = save_path
        else:
            filepath = filename

        # 发送 HTTP 请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        print(f"START DOWNLOAD: {filename}")
        print(f"SIZE: {total_size / (1024 * 1024):.2f} MB" if total_size else "SIZE: Unknown")

        # 写入文件并显示进度
        with open(filepath, 'wb') as f:
            downloaded = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                if total_size:
                    progress = downloaded / total_size * 100
                    print(f"Download Progress: {progress:.2f}% ({downloaded}/{total_size})", end='\r')

        print(f"The Video is saved in: {os.path.abspath(filepath)}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # 示例使用
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='./videos')
    parser.add_argument('--url_file', type=str,
                        default='./video_urls.json')

    args = parser.parse_args()

    all_videos = read_json(args.url_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for item in all_videos:
        download_video(item['url'], os.path.join(args.output_dir, item["target_name"]))