# train VQ model
bash scripts/tokenizer/train_vq.sh --image-size 256 --vq-model VQ-16 --data-path /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/dataset/imagenet_train_filelist.txt --epochs 40


# extract code
bash /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/scripts/autoregressive/extract_codes_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --dataset aoss \
    --data-path /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/dataset/imagenet_train_filelist.txt \
    --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode/imagenet_code_384_c2i_flip_ten_crop \
    --ten-crop --crop-range 1.1 --image-size 384

bash /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/scripts/autoregressive/extract_codes_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/pretrained_models/vq_ds16_c2i.pt \
    --data-path /mnt/afs/zhengmingkai/whl/llamagen/ILSVRC/Data/CLS-LOC/train \
    --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode2/imagenet_code_256_c2i_flip_ten_crop_105 \
    --ten-crop --crop-range 1.05 --image-size 256

bash /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/scripts/autoregressive/extract_codes_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --data-path /mnt/afs/zhengmingkai/whl/llamagen/ILSVRC/Data/CLS-LOC/train \
    --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode2/imagenet_code_256_c2i_flip_ten_crop \
    --ten-crop --crop-range 1.1 --image-size 256

bash /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/scripts/autoregressive/extract_codes_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --dataset aoss \
    --data-path /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/dataset/imagenet_train_filelist.txt \
    --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode2/imagenet_code_256_c2i_flip_ten_crop \
    --ten-crop --crop-range 1.1 --image-size 256

# training
torchrun \
    --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=29500 \
    /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/autoregressive/train/train_c2i.py \
    --gpt-model="GPT-B" \
    --image-size=256 \
    --downsample-size=16 \
    --global-batch-size=256 \
    --lr=1e-4 \
    --epochs=20 \
    --ema --no-compile \
    --results-dir="results_ref" \
    --cloud-save-path /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/cloud_save \
    --dataset="imagenet_code" \
    --code-path="/mnt/afs/zhengmingkai/zyr/ExtractedCode/imagenet_code_256_c2i_flip_ten_crop" 


torchrun \
    --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=29500 \
    /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/autoregressive/train/train_c2i.py \
    --gpt-model="GPT-B" \
    --gpt-ckpt="/mnt/afs/zhengmingkai/zyr/pretrained_models/c2i_B_256.pt" \
    --image-size=256 \
    --downsample-size=16 \
    --global-batch-size=256 \
    --lr=0.0001 \
    --epochs=20 \
    --no-compile \
    --is-lr-scheduler --warmup_percent 0.01 \
    --results-dir="results_ref" \
    --cloud-save-path /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/cloud_save \
    --dataset="imagenet_code" \
    --code-path="/mnt/afs/zhengmingkai/zyr/ExtractedCode/imagenet_code_256_c2i_flip_ten_crop" 


--gpt-ckpt="/mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/results_ref/001-GPT-B/checkpoints/0200000.pt"

torchrun \
    --nnodes=4 --nproc_per_node=8 --node_rank=3 \
    --master_addr=172.21.0.38 --master_port=29500 \
    /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/autoregressive/train/train_c2i.py \
    --gpt-model="GPT-B" \
    --gpt-ckpt="/mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/results_ref/001-GPT-B/checkpoints/0200000.pt" \
    --image-size=256 \
    --downsample-size=16 \
    --is-lr-scheduler --lr 4e-4 --max-lr 4e-4 --min-lr 1e-5 --warmup_percent 0.25 --const_percent 0 --cosine_percent 0.75 \
    --global-batch-size 2048 --epochs 400 \
    --no-compile \
    --results-dir="results_ref" \
    --cloud-save-path /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/cloud_save \
    --dataset="imagenet_code" \
    --code-path="/mnt/afs/zhengmingkai/zyr/ExtractedCode/imagenet_code_256_c2i_flip_ten_crop" 

# sampling
bash /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/c2i_XL_384.pt --gpt-model GPT-XL \
    --image-size 384 --image-size-eval 256 --cfg-scale 2.0 \
    --sample-dir samples --num-fid-samples 64 --per-proc-batch-size 8

bash /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/results_ref/017-GPT-B/checkpoints/1500000.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 \
    --sample-dir samples --num-fid-samples 800 --per-proc-batch-size 8

bash /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/c2i_XL_384.pt --gpt-model GPT-XL \
    --image-size 384 --image-size-eval 256 --cfg-scale 2.0 \
    --sample-dir samples --num-fid-samples 80 --per-proc-batch-size 8



# Evaluation
python3 evaluations/c2i/evaluator.py \
    /mnt/afs/zhengmingkai/zyr/LlamaGenOri/evaluations/VIRTUAL_imagenet256_labeled.npz \
    /mnt/afs/zhengmingkai/zyr/LlamaGen_ori2/samples/GPT-B-400e_LlamaGen_RARlr-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0.npz