import json

# 读取train.json文件
with open('/Users/luao/Downloads/data/processed_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 创建新字典来存储所有action
action_dict = {}

# 遍历训练数据，提取所有action
for item in train_data:
    if 'action' in item:
        action = item['action']
        # 使用action作为key，可以给每个action分配一个唯一的ID作为value
        if action not in action_dict:
            action_dict[action] = len(action_dict)

# 可以选择将action字典保存到文件中
with open('action_dict.json', 'w', encoding='utf-8') as f:
    json.dump(action_dict, f, ensure_ascii=False, indent=4)

print(f"总共收集到 {len(action_dict)} 个不同的action")
