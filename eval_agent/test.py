import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


import json
import time
import logging
from typing import List, Dict, Union, Any
import requests
from fastchat.model.model_adapter import get_conversation_template
from requests.exceptions import Timeout, ConnectionError
from fastchat.conversation import SeparatorStyle

messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]
controller_addr = 'http://localhost:21001'
worker_addr = 'http://localhost:21002'
if worker_addr == "":
    raise ValueError
gen_params = {
    "model": 'exp_traj-Phi-3-mini-4k-instruct-webshop-sft',
    "temperature": 1.0,
    "max_new_tokens": 512,
    "echo": False,
    "top_p": True,
    "logprobs": True,
    "top_logprobs": True,
}
conv = get_conversation_template('Phi-3-mini-4k-instruct')
for history_item in messages:
    role = history_item["role"]
    content = history_item["content"]
    if role == "user":
        conv.append_message(conv.roles[0], content)
    elif role == "assistant":
        conv.append_message(conv.roles[1], content)
    else:
        raise ValueError(f"Unknown role: {role}")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
if conv.sep_style == SeparatorStyle.PHI3:
    prompt = prompt.replace(conv.stop_str, '')

gen_params.update(
    {
        "prompt": prompt,
        "stop_token_ids": conv.stop_token_ids,
    }
)
headers = {"User-Agent": "FastChat Client"}
for _ in range(3):
    try:
        response = requests.post(
            controller_addr + "/worker_generate_stream",
            headers=headers,
            json=gen_params,
            stream=True,
            timeout=120,
        )
        text = ""
        for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if line:
                data = json.loads(line)
                if data["error_code"] != 0:
                    assert False, data["text"]
                text = data["text"]
        print(text)
    # if timeout or connection error, retry
    except Timeout:
        print("Timeout, retrying...")
    except ConnectionError:
        print("Connection error, retrying...")
    time.sleep(5)
else:
    raise Exception("Timeout after 3 retries.")