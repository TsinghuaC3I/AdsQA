<p align="center">
    <img src="./assets/project-name.png" alt="Project cover image"/> <br />
</p>


*Keyworks: Advertisement Videos, VideoQA, Multimodal Reasoning, GRPO*

🎉 This work has been accepted by ICCV 2025.  
:triangular_flag_on_post: Its arXiv version is available at [https://arxiv.org/abs/2509.08621](https://arxiv.org/abs/2509.08621) ![](https://img.shields.io/badge/abs-2025.09-red).

---

  Xinwei Long<sup>1*</sup>, Kai Tian<sup>1*</sup>, Peng Xu<sup>1+</sup>, Guoli Jia<sup>1</sup>, Jingxuan Li<sup>2</sup>, Sa Yang<sup>3</sup>, Yihua Shao<sup>4</sup>, Kaiyan Zhang<sup>1</sup>,Che Jiang<sup>1</sup>, Hao Xu<sup>5</sup>, Yang Liu<sup>2</sup>, Jiaheng Ma<sup>2</sup>,
  Bowen Zhou<sup>1,6†</sup>

<sup>1</sup> Tsinghua University

<sup>2</sup> Independent Researcher

<sup>3</sup> Peking University

<sup>4</sup> CASIA

<sup>5</sup> Harvard University

<sup>6</sup> Shanghai Artificial Intelligence Lab

<sup>*</sup> Equal Contribution

<sup>+</sup> corresponding authors

<a href="https://arxiv.org/abs/2509.08621"><img src='https://img.shields.io/badge/arXiv-CDP-red' alt='Paper PDF'></a>	<a href='https://tsinghuac3i.github.io/AdsQA/'><img src='https://img.shields.io/badge/Project_Page-CDP-green' alt='Project Page'></a>

---

AdsQA is the **first large-scale benchmark** targeting advertisement video understanding through **LLMs**. Ad videos are rich, symbolic, emotionally charged, and ideal for evaluating cognitive-level reasoning beyond physical perception. 

- **🌟 Why ads?** Unlike typical visual data, ads are professionally crafted to convey themes, metaphors, and targeted emotions.
- **📦 What’s AdsQA?** A benchmark built on 1,544 ad videos and 10,962 clips totaling 22.7 hours, annotated via a novel multi-agent pipeline.
- **🚀 Our Model: ReAd-R** is a Reinforced Ad Reasoner trained using reward-based optimization, outperforming chain-of-thought and agent-based methods.
- **🎯 5 Tasks**: Visual Concepts, Emotion, Themes, Persuasion, and Audience.

---

🔥 AdsQA is used as the test set of [ICCV 2025 MARS2 multimodal reasoning challenge](https://mars2workshop.github.io/iccv2025/). 

💥 See the MARS2 official report at [https://arxiv.org/abs/2509.14142](https://arxiv.org/abs/2509.14142) ![](https://img.shields.io/badge/abs-2025.09-red).

---

<img align="center" src="./assets/Figure 1.png" >


- **Our Contribution.** 

  - The AdsQA benchmark introduces a comprehensive, large-scale video QA dataset specifically designed around the complex and information-rich nature of advertisement videos. It offers a diverse and well-structured data source to evaluate LLMs on implicit reasoning tasks.

  <p align="center">
      <img src="./assets/Figure 2.png" width="800px"/> <br />
      <em>Figure: Statistics of AdsQA benchmark (duration, domain, regions, etc).</em>
  </p>


  - ReAd-R. We propose ReAd-R—a DeepSeek-R1–styled RL reasoning model that reflects, answers, and learns from outcome-based rewards, avoiding costly step-wise/COT supervision. 
  <p align="center">
    <img src="./assets/Figure 3.png" width="520px"/> <br />
    <em>Figure: Architecture of ReAd-R.</em>
</p>

- **Experiments** 
<p align="center">
    <img src="./assets/table 1.png" width="800px"/> <br />
    <!-- <em>Figure: Sample predictions vs ground-truth across models.</em> -->
</p>







# Get Start

## Data Acquisition

#### 1. Video Data Acquisition

According to the Terms of Use of the data source, we cannot store or redistribute the original video files. Instead, we provide open-source access to the video URLs. Please follow these steps to acquire the video data:

   - Obtain the complete list of video URLs from [this link](https://huggingface.co/datasets/TsinghuaC3I/AdsQA/blob/main/video_urls.json). The file contains URLs for both the training and test set videos.
   - Use our provided script `preprocess/download_videos.py` to download all videos.
   - Example usage:
     ```bash
     python preprocess/download_videos.py --url_file [path_to_url_file] --output_dir [video_output_directory]
     ```
     
   **If any videos are inaccessible or the URLs have expired, please feel free to open an issue or contact us directly via email.**
   
#### 2. Video Preprocessing (Optional):
   - For our ReAd-R model, we preprocessed videos using `video_clip.py` and `preprocess/transform_parquet.py`. Preprocessed files are also available for convenience at [this link](https://huggingface.co/datasets/TsinghuaC3I/AdsQA/tree/main/processed_videos).
   - Example usage:
     ```bash
     cd preprocess
     python video_clip.py # 
     python transform_parquet.py # converts the dataset into Parquet format for training.
     ```
   - Note: You may customize preprocessing (e.g., different sampling rates, resolutions) based on your specific requirements.


#### 3. Question and Annotation Data Acquisition

Download the following annotation files from [this link](https://huggingface.co/datasets/TsinghuaC3I/AdsQA/tree/main):

- `train.json` - Training set questions and annotations
- `testset_question.json` - Test set (ids, videos, and questions) for inference
- `testset_groundtruth.json` - Test set (ids, questions, ground-truth answers, and their meta_info) for model-based auto evaluation.

**!!! Important Usage Note: The meta_info field is only for model-based auto evaluation purposes; DO NOT use meta_info as model input during the inference.**

## Training, Inference, and Evaluation

#### 1. Requirements
We use the **[EasyR1](https://github.com/hiyouga/EasyR1)** framework for reinforcement learning (RL) training.

```bash
conda create -n ReadR python=3.10
conda activate ReadR

cd ReadR
pip install -e .
```

#### 2. Train

We provide the training code for ReAd-R. Please use the following script to run the training code.

~~~
bash examples/adsqa.sh
~~~

Meanwhile, we have released our checkpoint (**Model files re-uploaded on 2025-10-22**):

|         Model          |                     LINK                     |
| :--------------------: | :--------------------------------------------------: |
|  Qwen2.5-7B-VL-ReAd-R  | 🤗[Huggingface](https://huggingface.co/TsinghuaC3I/Qwen2.5-7B-VL-ReAd-R) |


#### 3. Inference

We provide inference scripts in the `evaluation` directory. During inference, we use Automatic Speech Recognition (ASR) results as input features. The corresponding ASR data (`asr_set.json`) is also included in the same directory. The ASR transcripts were generated using Whisper and then translated into English via GPT-4o. For visual processing, we extract frames at 1 FPS (with a maximum of 32 frames) and use the default max_pixel setting from Qwen2.5-VL.

This script will create a directory for each question (named with its question_id), inside which the prediction result from each model will be saved using the given model name (See examples in `./evaluation/results`). Please use the following script to run the inference code.

~~~
# for GRPO model inference
python evaluation/eval_adQA.py --video_dir your_video_path --file_dir test_file_dir --model_dir test_model_save_path --model_name your_model_name
# for Qwen2.5-VL model inference
python evaluation/eval_adQA_qwen2d5-7b.py --video_dir your_video_path --file_dir test_file_dir --model_dir test_model_save_path --model_name your_model_name
~~~
#### 4. Evaluation

We use GPT-4o as the judge model. Please refer to our `./evaluation/model_evaluation.py` script to score the prediction results. In this file, you will need to specify the prediction file name, directory, and your GPT-4o API key and base URL. 
The model-based evaluation results will be saved to the `score` field in the prediction file.

Please use the following script to run the model-based evaluation code.

~~~
# for model-based inference
python evaluation/model_evaluation.py --eval_name prediction_file_name --test_file groundtruth_file --results_dir dir_you_save_prediction_files --api_key your_api_key --api_base your_api_url_base
~~~

## Contact

If you have any questions, please feel free to contact me:

longxw22@mails.tsinghua.edu.cn

tk23@mails.tsinghua.edu.cn

## :star: Citation
If you find our dataset, code, or model useful in your research, please consider citing our paper and MARS2 Workshop:

    ```
    @inproceedings{long2025adsqa,
        author    = {Long, Xinwei and Tian, Kai and Xu, Peng and Jia, Guoli and Li, Jingxuan and Yang, Sa and Shao, Yihua and Zhang, Kaiyan and Jiang, Che and Xu, Hao and Liu, Yang and Ma, Jiaheng and Zhou, Bowen},
        title     = {AdsQA: Towards Advertisement Video Understanding},
        booktitle = {ICCV},
        year      = {2025}
    }
    ```
    
    ```
    @inproceedings{xu2025mars2,
    author    = {Xu, Peng and Xiong, Shengwu and Zhang, Jiajun and Chen, Yaxiong and Zhou, Bowen and Loy, Chen Change and Clifton, David and Lee, Kyoung Mu and Van Gool, Luc and others},
    title     = {MARS2 2025 Challenge on Multimodal Reasoning: Datasets, Methods, Results, Discussion, and Outlook},
    booktitle = {ICCV Workshop},
    year      = {2025}
    }
    ```
