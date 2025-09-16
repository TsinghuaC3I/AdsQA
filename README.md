<p align="center">
    <img src="./assets/project-name.png" alt="Project cover image"/> <br />
</p>


*Keyworks: Advertisement Videos, VideoQA, Multimodal Reasoning, GRPO*

---

  Xinwei Long<sup>1*</sup>, Kai Tian<sup>1*</sup>, Peng Xu<sup>1</sup>, Guoli Jia<sup>1</sup>, Jingxuan Li<sup>2</sup>, Sa Yang<sup>3</sup>, Yihua Shao<sup>4</sup>, Kaiyan Zhang<sup>1</sup>,Che Jiang<sup>1</sup>, Hao Xu<sup>5</sup>, Yang Liu<sup>2</sup>, Jiaheng Ma<sup>2</sup>,
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

Meanwhile, we have released our checkpoint:

|         Model          |                     LINK                     |
| :--------------------: | :--------------------------------------------------: |
|  Qwen2.5-7B-VL-ReAd-R  | 🤗[Huggingface](https://huggingface.co/TsinghuaC3I/Qwen2.5-7B-VL-ReAd-R) |


#### 3. Inference


#### 4. Evaluation


## Contact

If you have any questions, please feel free to contact me:

longxw22@mails.tsinghua.edu.cn

tk23@mails.tsinghua.edu.cn

## Citation
If you find our dataset, code, or model useful in your research, please consider citing our work:
```
@misc{long2025adsqaadvertisementvideounderstanding,
      title={AdsQA: Towards Advertisement Video Understanding}, 
      author={Xinwei Long and Kai Tian and Peng Xu and Guoli Jia and Jingxuan Li and Sa Yang and Yihua Shao and Kaiyan Zhang and Che Jiang and Hao Xu and Yang Liu and Jiaheng Ma and Bowen Zhou},
      year={2025},
      eprint={2509.08621},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.08621}, 
}
```
