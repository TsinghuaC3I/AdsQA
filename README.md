<!--  AdsQA 
# [ICCV 2025] AdsQA: Towards Advertisement Video Understanding -->
<p align="center">
    <img src="./assets/project-name.png" alt="Project cover image"/> <br />
</p>


<div align="center">
<!-- 🔍 <b>Paper</b>: <i><a href="#">Coming Soon</a></i> | 
  📊 <b>Benchmark</b>: <i>Coming Soon</i> |
  🤖 <b>Model</b>: <i>Coming Soon</i> 
-->   
    <p align="center">
        <i>Keywords: Advertisement Videos, VideoQA, Multimodal Reasoning, GRPO</i>
    </p>
</div>

----
<img align="center" src="./assets/video.png" >

<img align="center" src="./assets/Figure 1.png" >

AdsQA is the **first large-scale benchmark** targeting advertisement video understanding through **LLMs**. Ad videos are rich, symbolic, emotionally charged, and ideal for evaluating cognitive-level reasoning beyond physical perception. 

- **🌟 Why ads?** Unlike typical visual data, ads are professionally crafted to convey themes, metaphors, and targeted emotions.
- **📦 What’s AdsQA?** A benchmark built on 1,544 ad videos and 10,962 clips totaling 22.7 hours, annotated via a novel multi-agent pipeline.
- **🚀 Our Model: ReAd-R** is a Reinforced Ad Reasoner trained using reward-based optimization, outperforming chain-of-thought and agent-based methods.
- **🎯 5 Tasks**: Visual Concepts, Emotion, Themes, Persuasion, and Audience.



## 📐 Tasks Overview
1. **Visual Concept Understanding (VU)** – Identify scenes, characters, symbols.
2. **Emotion Recognition (ER)** – Infer emotional undertones and user impact.
3. **Theme Extraction (TE)** – Distill core message and implied storytelling.
4. **Persuasion Strategy (PS)** – Decode rhetorical and marketing tactics.
5. **Audience Modeling (AM)** – Predict target demographics and profiles.

## 📊 Dataset Details

<p align="center">
    <img src="./assets/Figure 2.png" width="800px"/> <br />
    <em>Figure: Statistics of AdsQA benchmark (duration, domain, regions, etc).</em>
</p>

- 1,544 ad videos from 9 domains and 6 continents.
- 7,859 open-ended QA pairs across 5 categories.
- Clips sampled and described using multi-modal pipeline (frames, ASR, descriptions).


## 🧠 ReAd-R: Reinforced Ad Reasoner

<p align="center">
    <img src="./assets/Figure 3.png" width="520px"/> <br />
    <em>Figure: Architecture of ReAd-R.</em>
</p>

ReAd-R simulates **human-like heuristic reasoning** via reinforcement learning (RL):

- Reward-driven answer generation with no chain-of-thought templates.
- Optimized with GRPO using self-generated reward signals.
- Supports open-ended VideoQA and is base-model agnostic.

## 🧪 Experiments

<p align="center">
    <img src="./assets/table 1.png" width="800px"/> <br />
    <!-- <em>Figure: Sample predictions vs ground-truth across models.</em> -->
</p>

### Key Findings:
- AdsQA is substantially harder than typical video QA benchmarks.
- ReAd-R excels at **implicit logic** and **persuasive reasoning**.
- Chain-of-thought and multi-agent search show limited generalization.

## 🧪 Case Studies

<p align="center">
    <img src="./assets/Figure 4.png" width="520px"/> <br />
    <!-- <em>Figure: Sample predictions vs ground-truth across models.</em> -->
</p>




## 🏋️ Training (ReAd-R via EasyR1)

We use the **[EasyR1](https://github.com/hiyouga/EasyR1)** framework for reinforcement learning (RL) training.

```bash
# From the project root
cd code/train/EasyR1
pip install -e .
bash examples/adsqa.sh
```
