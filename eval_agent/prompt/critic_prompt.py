import json
import re

def get_critic_prompt():
    FEW_SHOT_EXAMPLES = ''''''
    with open('eval_agent/prompt/icl_examples/webshop_icl.json', 'r', encoding='utf-8') as file:
        conversations = json.load(file)[0]
        tmp_conv = ''
        for i, conv in enumerate(conversations):
            role = conv['role']
            content = conv['content']
            tmp_conv += content + '\n'
            if i % 2 == 1:
                matches = re.findall(r'Thought: (.*?)\nAction: ', tmp_conv)
                if matches:
                    last_thought = matches[-1]
                else:
                    raise ValueError(tmp_conv)
                FEW_SHOT_EXAMPLES += tmp_conv.rsplit('Thought: ', 1)[0] + f'Next step plan: {last_thought}\n\n'

    FEW_SHOT_EXAMPLES = FEW_SHOT_EXAMPLES[:-2]

    prompt =  f'''
---
You will be given the history of a past experience in which you were placed in an environment and given a task to complete.
Do not summarize your environment and give the overall plan, but rather devise a specific, concise next step plan to attempt to complete the task.
There are four examples below:
{FEW_SHOT_EXAMPLES}
---
Now, it's your turn and here is the the history of a past experience.
Observation:
'''
    return prompt

