import requests
import json
import time
from prompt.critic_prompt import get_critic_prompt
from requests.exceptions import Timeout, ConnectionError


def send_agent_request(url, params):
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

prompt = '''
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

---
You will be given an instruction which specifies a task to be completed, and the history of a past experience in which you were placed in an environment.
Based on this given information, now you have a preliminary thought for planning to do next, but it may not be well thought out.
You need to further refine your ideas to facilitate the completion of the task.

Input format:
...(Past experience: Thought & Action & Observation can repeat n times)
Thought: <Thought>

Output format:
Refined Thought: <your plan>

There are two examples below:

example 1:
Observation:
WebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Search
Thought: I think I should use the search bar to look for the product I need.
Refined Thought: I think I should use 'l'eau d'issey 6.76 fl oz bottle price < 100.00' as the keyword and enter it into the search bar to look for the product I need.

example 2:
Observation:
WebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Search
Thought: I think I should use the search bar to look for the product I need.
Action: search[l'eau d'issey 6.76 fl oz bottle price < 100.00]
Observation:
Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L'eau d'Issey by Issey Miyake for Women 3.3 oz Eau de Toilette Spray [SEP] $49.98 [SEP] B0012S249E [SEP] L'eau D'issey By Issey Miyake For Women. Shower Cream 6.7-Ounces [SEP] $31.36 [SEP] B01H8PGKZS [SEP] L'eau D'Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [SEP] $67.97 [SEP] B00G3C8FHE [SEP] L'Eau d'Issey pour Homme - Eau de Toilette 4.2 fl oz [SEP] $51.25 [SEP] B000R94HRG [SEP] Issey Miyake L'Eau D'Issey Pour Homme Eau De Toilette Natural Spray [SEP] $44.99 [SEP] B000C214CO [SEP] Issey Miyake L'eau D'issey Eau de Toilette Spray for Men, 4.2 Fl Oz [SEP] $53.99 [SEP] B0018SBRDC [SEP] Issey Miyake L'eau d'Issey for Women EDT, White, 0.84 Fl Oz [SEP] $27.04 [SEP] B000XEAZ9Y [SEP] L'eau De Issey By Issey Miyake For Men. Eau De Toilette Spray 6.7 Fl Oz [SEP] $67.08 [SEP] B079HZR2RX [SEP] L'eau d'Issey Pure by Issey Miyake for Women 3.0 oz Nectar de Parfum Spray [SEP] $71.49
Thought: I think I should click on the product I need, which is B000VOHH8I.
Refined Thought: Compared with other products, B000VOHH8I can comply with user instruction better. I think I should click on the product B000VOHH8I.

Now, it's your turn and here is the the history of a past experience and current thought.
Observation:
WebShop [SEP] Instruction: [SEP] i am looking for some 18 inch pearl platinum synthetic hair extensions, and price lower than 30.00 dollars [SEP] Search<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Thought: I think I should use the search bar to look for the product I need.
'''
new_stop = {'\n---', '\nObservation:', '\nTask:', '<|eot_id|>'}
controller_addr = 'http://localhost:21001'
gen_params = {
    "model": 'Meta-Llama-3.1-8B-Instruct',
    "temperature": 0.0,
    "max_new_tokens": 512,
    "echo": False,
    "prompt": prompt,
    "stop": list(new_stop),
    "stop_token_ids": [128001, 128009],
}
prompt_critic = get_critic_prompt()
text = send_agent_request(controller_addr, gen_params)
print(text)