#!/bin/bash

# Activate Python virtual environment
source venv/bin/activate

# Set paths
pretrained_model="/path/to/directory/chilloutmix_NiPrunedFp32Fix.safetensors" # Base model path
train_data_dir="/path/to/directory"
output_dir="./output"
logging_dir="./logs"

# Training parameters
resolution="768,768"
batch_size=2
max_train_epochs=10
save_every_n_epochs=1
network_dim=64
network_alpha=32
clip_skip=2
train_unet_only=0
train_text_encoder_only=0

# Learning rates
lr="5e-5"
unet_lr="5e-5"
text_encoder_lr="6e-6"
lr_scheduler="cosine_with_restarts"
lr_warmup_steps=50
lr_restart_cycles=1

# Output settings
output_name="yujinive_v2"
save_model_as="safetensors"

# Bucket resolution
min_bucket_reso=64
max_bucket_reso=768  # Adjust to the maximum resolution

# Run training
python "./sd-scripts/train_network.py" \
--enable_bucket \
--pretrained_model_name_or_path="$pretrained_model" \
--train_data_dir="$train_data_dir" \
--output_dir="$output_dir" \
--resolution="$resolution" \
--network_module=networks.lora \
--max_train_epochs="$max_train_epochs" \
--learning_rate="$lr" \
--unet_lr="$unet_lr" \
--text_encoder_lr="$text_encoder_lr" \
--lr_scheduler="$lr_scheduler" \
--lr_warmup_steps="$lr_warmup_steps" \
--lr_scheduler_num_cycles="$lr_restart_cycles" \
--network_dim="$network_dim" \
--network_alpha="$network_alpha" \
--output_name="$output_name" \
--train_batch_size="$batch_size" \
--save_every_n_epochs="$save_every_n_epochs" \
--mixed_precision="fp16" \
--save_precision="fp16" \
--seed="1337" \
--cache_latents \
--clip_skip="$clip_skip" \
--prior_loss_weight=1 \
--max_token_length=225 \
--caption_extension=".txt" \
--save_model_as="$save_model_as" \
--min_bucket_reso="$min_bucket_reso" \
--max_bucket_reso="$max_bucket_reso" \
--xformers \
--shuffle_caption

echo "Training completed"
read -n 1 -s -r -p "Press any key to continue..."