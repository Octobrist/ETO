import os
import json
import re
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_distribution(data, bins=20, title="Data Distribution", xlabel="Values", ylabel="Frequency"):
    plt.figure(figsize=(8, 6))  # 设置图形大小

    # 绘制直方图
    plt.hist(data, bins=bins, color='blue', edgecolor='black', alpha=0.7)

    # 设置图表标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()

def load_all_json_files(directory):
    json_data = {}
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                # 打开并读取 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['id'] = filename.replace('.json','')
                    json_data[data['id']] = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return json_data

def write_to_json_file(data, file_path):
    try:
        # 打开文件以写入模式 ("w")
        with open(file_path, 'a', encoding='utf-8') as f:
            # 使用 json.dump 将数据写入文件
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")


directory_path = "/home/huan/projects/works/ETO/outputs/gpt-4/webshop"  # 你的json文件目录路径
json_data1 = load_all_json_files(directory_path)

directory_path = "/home/huan/projects/works/ETO/outputs/llama3-8b/webshop"  # 你的json文件目录路径
json_data2 = load_all_json_files(directory_path)

avg_reward1 = 0
total_num1 = 0
success_num1 = 0
avg_step = 0
reward_data1 = []

avg_reward2 = 0
total_num2 = 0
success_num2 = 0
reward_data2 = []

for id, data1 in tqdm(json_data1.items()):
    data2 = json_data2[id]
    meta1 = data1['meta']
    meta2 = data2['meta']
    conv1 = data1['conversations'][10:]
    conv2 = data2['conversations'][10:]

    # if isinstance(meta1['reward'], float):
    #     reward_data1.append(meta1['reward'])
    #     avg_reward1 += meta1['reward']
    # total_num1 += 1
    # if meta1['success'] == True:
    #     success_num1 += 1

    # if isinstance(meta2['reward'], float):
    #     avg_reward2 += meta2['reward']
    # total_num2 += 1
    # if meta2['success'] == True:
    #     success_num2 += 1

    for conv in conv1:
        if 'Action: click[< Prev]' in conv['value'] or 'Action: click[Next >]' in conv['value']:
            if isinstance(meta1['reward'], float):
                reward_data1.append(meta1['reward'])
                avg_reward1 += meta1['reward']
            else:
                reward_data1.append(0)
            if meta1['success'] == True:
                success_num1 += 1
            total_num1 += 1

    # if isinstance(meta2['reward'], float) and meta1['reward'] == None:
    # if isinstance(meta2['reward'], float) and isinstance(meta1['reward'],float) and meta1['reward'] < meta2['reward']:
    #     conv1 = data1['conversations'][10:]
    #     conv2 = data2['conversations'][10:]
    # print(1)
    # ori_ins, atk_ins = attack_ins(conv[-step_num*2-1]['value'])
    # ori_atk_pair[data['id']] = (ori_ins, atk_ins)
print(avg_reward1/total_num1)
print(total_num1, success_num1)

# print(avg_reward2/total_num2)
# print(total_num2, success_num2)
plot_distribution(reward_data1)