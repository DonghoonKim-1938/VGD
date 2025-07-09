DEVICES=0

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
  --nproc_per_node=1 \
  --master_port=1235 \
  ./vgd/inference.py \
  --config=./vgd/config/vgd.yaml \
  "wandb.mode=disabled" \
  "data_dir=['/datasets/lexica.art/images/0.png']" \
  "seed=0" \
  "wandb.mode=disabled" \
  "dataset_name=prompt_inversion" \
  "save_dir=./prompt_inversion/" \
  "model.max_length=77" \
  "model.min_length=60" \

