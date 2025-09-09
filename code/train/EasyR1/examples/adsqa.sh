MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

FORMAT_PROMPT="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here.</answer>."""

python3 -m verl.trainer.main \
    config=examples/adsqa.yaml \
    data.train_files=dataset/adv/adsqa.parquet \
    data.val_files=dataset/adv/adsqa.parquet \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=adsqa \
    trainer.n_gpus_per_node=8 \
    worker.rollout.limit_images=8

