#!/bin/bash
model_list=("exp_traj-Phi-3-mini-4k-instruct-webshop-sft" "Mistral-7B-Instruct-v0.2" "llama3-8b" "Phi-3-mini-4k-instruct" "gpt-3.5-turbo" "gpt-4")

for item in "${model_list[@]}"; do
    if [[ $item == *gpt* ]]; then
      echo $item
      python -m eval_agent.main --agent_config openai --model_name $item --exp_config webshop --split test --verbose
    else
      echo /home/huan/projects/llm/$item
      python -m fastchat.serve.model_worker --model-path /home/huan/projects/llm/$item --port 21002 --worker-address http://localhost:21002 &
      pid=$!
      sleep 120

      python -m eval_agent.main --agent_config fastchat --model_name $item --exp_config webshop --split test --verbose &
      pid2=$!
      wait $pid2

      kill $pid
    fi
done