# HPRO: Hierarchical Preference Ranking Optimization

基于大语言模型的销售线索意向预测，采用层级化偏好排序优化。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练
deepspeed --num_gpus=8 src/train.py \
    --deepspeed_config cfg/ds_config_bf16_stage2.json \
    --data_file data/your_data.pkl \
    --pretrained_model /path/to/Qwen1.5-1.8B \
    --out_dir output/exp1 \
    --lora_r 16 \
    --use_hpro \
    --batch_size 32 \
    --epochs 1
```


### 漏斗层级
```
Stage 3: Lock-in (订单锁定)    >
Stage 2: Test Drive (试驾)     >
Stage 1: Call (电话接通)       >
Stage 0: Defeat (战败流失)
```



## 数据格式

支持两种格式（自动检测）：

**基础格式**（使用标准pairwise）：
- 包含字段：`corpus`, `lock_label`, `lock_label_str`

**HPRO格式**（启用层级优化）：
- 额外提供：`funnel_stage` (0-3)
  - 0: 战败, 1: 电话, 2: 试驾, 3: 锁单

无漏斗数据时，HPRO自动降级到pairwise loss。

## License

MIT
