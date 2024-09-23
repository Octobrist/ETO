model_name=Phi-3-mini-4k-instruct
task=$1

exp_name=$2

node_num=1  # number of GPUs
num_workers=4   # number of inference workers

model_path=$3 # path to the original LLM
save_dir=$4    # checkpoint save path

# Part 1: SFT stage
sft_data_path="data/${task}_sft.json"
batch_size=64
micro_batch_size=2
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

sft_model_name=${exp_name}-${model_name}-${task}-sft

python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20001 fastchat/train/train.py \
    --model_name_or_path ${model_path}${model_name} \
    --data_path ${sft_data_path} \
    --bf16 True \
    --output_dir ${save_dir}${sft_model_name} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Phi3DecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess False

# if failed, exit
if [ $? -ne 0 ]; then
    echo "SFT training failed"
    exit 1
fi

# launch the FastChat controller
#python -u -m fastchat.serve.controller >> logs/${exp_name}-controller.log 2>&1 &
#fs_controller_pid=$!
#
## Evaluate the base agent
#fs_worker_port=21002
#CUDA_VISIBLE_DEVICES=0 python -u -m fastchat.serve.model_worker --model-path ${save_dir}${sft_model_name} --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> logs/${exp_name}-model_worker.log 2>&1 &
#
#fs_worker_pid=$!
#sleep 60
#
## evaluate on the test set
#python -m eval_agent.main --agent_config fastchat --model_name ${sft_model_name} --exp_config ${task} --split test
#
## if failed, exit
#if [ $? -ne 0 ]; then
#    echo "base agent evaluation failed"
#    kill -9 $fs_worker_pid
#    exit 1
#fi
## kill the model worker
#kill -9 $fs_worker_pid