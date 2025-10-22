MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path


export WANDB_API_KEY=""

# FORMAT_PROMPT="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here.</answer>."""
FORMAT_PROMPT="""You are permitted to gather any relevant clues, think step by step, to answer this question. Output your thought process (no length limit) and the final answer (using approximately 30 words; longer answers will be truncated) in the following format\n <think>your_thinking</think> <answer>your_answer_within_30_words</answer>"""
python3 -m verl.trainer.main \
    config=examples/adsqa.yaml \
    data.train_files=data/adsqa.parquet \
    data.val_files=data/adsqa.parquet \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=adsqa \
    trainer.n_gpus_per_node=4 \
    worker.rollout.limit_images=8

