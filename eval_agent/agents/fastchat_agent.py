import json
import time
import logging
import copy
from typing import List, Dict, Union, Any
import requests
from fastchat.model.model_adapter import get_conversation_template
from requests.exceptions import Timeout, ConnectionError
from fastchat.conversation import SeparatorStyle
from eval_agent.prompt.critic_prompt import get_critic_prompt
from .base import LMAgent

logger = logging.getLogger("agent_frame")


def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)



class FastChatAgent(LMAgent):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)
        self.controller_address = config["controller_address"]
        self.model_name = config["model_name"]
        self.critic_model_name = config["critic_model_name"]
        self.temperature = config.get("temperature", 0) # huan
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.top_p = config.get("top_p", 0)

    def get_conv_and_prompt(self, messages, model_name):
        conv = get_conversation_template(model_name)
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
        return conv, prompt

    def send_agent_request(self, url, params):
        headers = {"User-Agent": "FastChat Client"}
        for _ in range(3):
            try:
                # llm_logprobs = []
                response = requests.post(
                    url + "/worker_generate_stream",
                    headers=headers,
                    json=params,
                    stream=True,
                    timeout=120,
                )
                text = ""
                # data_logprobs = {'token':'', 'logprob':None}
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if line:
                        data = json.loads(line)
                        if data["error_code"] != 0:
                            assert False, data["text"]
                        text = data['text']
                return text
                # data_logprobs = data["logprobs"]
                # for i in range(len(data_logprobs['tokens'])):
                #     llm_logprobs.append({'token':data_logprobs['tokens'][i], 'logprob':data_logprobs['token_logprobs'][i]})
                # llm_logprobs.append({'token'})
                # return text, llm_logprobs
            # if timeout or connection error, retry
            except Timeout:
                print("Timeout, retrying...")
            except ConnectionError:
                print("Connection error, retrying...")
            time.sleep(5)
        else:
            raise Exception("Timeout after 3 retries.")

    def __call__(self, messages: List[dict]):
        controller_addr = self.controller_address
        worker_addr = controller_addr
        if worker_addr == "":
            raise ValueError
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            # "top_p": self.top_p,
            # "logprobs": True,
        }
        conv, prompt = self.get_conv_and_prompt(messages, self.model_name)
        stop_words = set()
        _add_to_set(self.stop_words, stop_words)
        _add_to_set(conv.stop_str, stop_words)
        # 1. generate ori thought
        gen_params.update(
            {
                "prompt": prompt,
                "stop": list(stop_words),
                "stop_token_ids": conv.stop_token_ids,
            }
        )
        ori_text = self.send_agent_request(controller_addr, gen_params)
        # 2. generate refined thought
        messages.append({
            'role': 'assistant',
            'content': f"{ori_text}"
        })
        critic_message = copy.deepcopy(messages[10:])
        critic_message[0]['content'] = get_critic_prompt() + critic_message[0]['content']
        critic_conv, critic_prompt = self.get_conv_and_prompt(critic_message, self.critic_model_name)
        stop_words = set()
        _add_to_set(self.stop_words, stop_words)
        _add_to_set('<|end|>', stop_words)
        _add_to_set(critic_conv.stop_str, stop_words)
        gen_params.update(
            {
                "prompt": critic_prompt,
                "stop": list(stop_words),
                "stop_token_ids": critic_conv.stop_token_ids,
            }
        )
        critic_text = self.send_agent_request(controller_addr, gen_params)
        messages.append({
            'role': 'assistant',
            'content': f"{critic_text}"
        })
        conv, prompt = self.get_conv_and_prompt(messages, self.model_name)
        # prompt = prompt.rsplit('<|eot_id|><|start_header_id|>assistant<|end_header_id|>', 1)[0]
        stop_words = set()
        _add_to_set(self.stop_words, stop_words)
        _add_to_set(conv.stop_str, stop_words)
        gen_params.update(
            {
                "prompt": prompt,
                "stop": list(stop_words),
                "stop_token_ids": conv.stop_token_ids,
            }
        )
        text = self.send_agent_request(controller_addr, gen_params)
        return text

