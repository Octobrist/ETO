import json
import re

def get_critic_prompt():
    # FEW_SHOT_EXAMPLES = ''''''
    # with open('eval_agent/prompt/icl_examples/webshop_icl.json', 'r', encoding='utf-8') as file:
    #     conversations = json.load(file)[0]
    #     tmp_conv = ''
    #     example_idx = 1
    #     for i, conv in enumerate(conversations):
    #         role = conv['role']
    #         content = conv['content']
    #         tmp_conv += content + '\n'
    #         if i == 1 or i == 3:
    #             matches = re.findall(r'Thought: (.*?)\nAction: ', tmp_conv)
    #             if matches:
    #                 last_thought = matches[-1]
    #             else:
    #                 raise ValueError(tmp_conv)
    #             FEW_SHOT_EXAMPLES += f"example {example_idx}:\n" + tmp_conv.rsplit('Thought: ', 1)[0] + f'Next plan: {last_thought}\n\n'
    #             example_idx += 1
    # FEW_SHOT_EXAMPLES = FEW_SHOT_EXAMPLES[:-2]

    prompt =  f'''
---
You will be given an instruction which specifies a task to be completed, and the history of a past experience in which you were placed in an environment.
Based on this given information, now you have a preliminary thought for planning to do next, but it may or may not be well thought out. 
You don't have to offer solutions, but you can offer some critical thinking. Make sure your output is within 128 tokens.

Input format:
... (Past experience: Thought & Action & Observation can repeat n times)
Thought: <Thought>
Action: <Action>

Output format:
Critical Thought: <critical thinking>

There are two examples below:
example 1:
Observation:
WebShop [SEP] Instruction: [SEP] i need gluten free vegetarian smoked peppered bacon - 4 ounce (pack of 2), and price lower than 40.00 dollars [SEP] Search
Thought: I think I should use the search bar to look for the product I need.
Action: search[gluten free vegetarian smoked peppered bacon 4 ounce pack of 2]
Critical Thought: The search keyword - 'gluten free vegetarian smoked peppered bacon 4 ounce pack of 2' might be too specific, and some products that meet the user's needs may be overlooked. Next time, I will do search["vegetarian bacon"] for a broader search. I will check that the new results will fulfill the gluten free and 4 ounce pack of 2 constraints.

example 2:
Observation:
WebShop [SEP] Instruction: [SEP] i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars [SEP] Search
Thought: I think I should use the search bar to look for the product I need.
Action: search[dairy free and apple variety pack of chips]
Observation:
WebShop [SEP] Instruction: [SEP] i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] [B07HRFJWP8] [SEP] Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total) [SEP] $100.0 [SEP] [B01KMHY5PG] [SEP]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars) [SEP] $21.49 [SEP] [B008D2X8C4] [SEP] Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz) [SEP] $100.0 [SEP]
Thought: I think I should click on the product I need, which is B008D2X8C4.
Action: click[B008D2X8C4]
Critical Thought: I accidentally choose a product that was $100, which is more than my budget of $30. Next time, I will choose the product meet the dairy free and the $30 budget constraints. Either way, the initial search results were not good, so I may also choose to click the 'Back to Search' button to the search page.

---
Now, it's your turn and here is the the history of a past experience and the current thought and plan.
Observation:
'''
    return prompt


