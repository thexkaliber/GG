import json
import os

# 读取现有的JSON文件
def read_json_data(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 处理并保存数据
def process_and_save(input_json_path, output_json_path):
    # 读取输入数据
    input_data = read_json_data(input_json_path)
    
    # 创建输出数据结构
    processed_data = {}
    
    # 遍历所有数据项并按ID整理
    for id, data in input_data.items():
        processed_data[id] = {
            'consequence': data.get('consequence', ''),
            'reasoning': data.get('reasoning', '')
        }
    
    # 保存处理后的数据
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功处理 {len(processed_data)} 条数据并保存到 {output_json_path}")

# 合并数据并保存
def merge_and_save():
    # 读取两个JSON文件
    processed_train = read_json_data('processed_train.json')
    merged_data = read_json_data('merded_data.json')
    
    # 遍历processed_train中的每个条目
    for item in processed_train:
        item_id = item['id']
        
        # 如果在merged_data中找到对应的ID
        if item_id in merged_data:
            # 更新consequence和reasoning字段
            item['consequence'] = merged_data[item_id]['consequence']
            item['reasoning'] = merged_data[item_id]['reasoning']
    
    # 保存更新后的数据
    with open('processed_train_updated.json', 'w', encoding='utf-8') as f:
        json.dump(processed_train, f, ensure_ascii=False, indent=2)
    
    print(f"成功更新 processed_train.json 中的数据并保存到 processed_train_updated.json")

def merge_final_inferences():
    # 读取原始训练数据
    processed_train = read_json_data('processed_train.json')
    
    # 读取最终推理结果
    final_inferences = read_json_data('final_inferences.json')
    
    # 创建以game_state_id为键的字典方便快速查找
    inference_map = {item["game_state_id"]: item for item in final_inferences}
    
    # 遍历处理训练数据
    for train_item in processed_train:
        item_id = train_item["id"]
        if item_id in inference_map:
            # 更新consequence和reasoning字段
            inference = inference_map[item_id]
            train_item["consequence"] = inference["consequence"]
            train_item["reasoning"] = inference["reasoning"]
    
    # 保存到新文件
    new_filename = 'processed_train_final.json'
    with open(new_filename, 'w', encoding='utf-8') as f:
        json.dump(processed_train, f, ensure_ascii=False, indent=2)
    
    print(f"成功合并数据并保存到 {new_filename}，共处理 {len(processed_train)} 条数据")

# 执行合并操作
merge_final_inferences()