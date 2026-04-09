# 环境准备
source shared/scripts/setup_cluster.sh
bash shared/scripts/download_data.sh

# Step 1: Baseline（无需训练，直接跑）
python pipeline/baseline_inference.py \
    --data_dir /tmp/data/ \
    --output_dir results/baseline/

# Step 2: Main 训练（LoRA + head）
python pipeline/train_main.py \
    --data_dir /tmp/data/ \
    --output_dir ~/SharedFolder/MDAIE/group6/checkpoints/main/

# Step 3: Ablation 训练（head only）
python pipeline/train_ablation.py \
    --data_dir /tmp/data/ \
    --output_dir ~/SharedFolder/MDAIE/group6/checkpoints/ablation/

# Step 4: 对比评估
python courses/shared/eval.py \
    --data_dir /tmp/data/ \
    --output_dir results/

# Step 5: 同步结果到 SharedFolder
cp -r results/ ~/SharedFolder/MDAIE/group6/results/