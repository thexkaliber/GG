import json
import os

def process_game_data(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    skipped_count = 0
    
    # 处理每个游戏状态
    for game_idx, game_sequence in enumerate(data):
        for state_idx, state in enumerate(game_sequence):
            try:
                # 检查action是否存在
                action = state.get('state', {}).get('walkthrough_act', '')
                if not action:
                    print(f"警告: 在游戏序列 {game_idx}, 状态 {state_idx} 中发现缺失的action")
                    print(f"状态数据: {json.dumps(state, ensure_ascii=False, indent=2)}")
                    skipped_count += 1

                formatted_state = {
                    "id": f"game_{game_idx}_state_{state_idx}",
                    "action": action,
                    "context": {
                        "location_name": state['state']['location']['name'],
                        "location_desc": state['state']['loc_desc'],
                        "surroundings": {
                            "objects": {}
                        },
                        "inventory": {
                            "desc": state['state']['inv_desc'],
                            "objects": {},
                            "attrs": state['state'].get('inv_attrs', [])  # 使用get方法以防inv_attrs不存在
                        }
                    },
                    "consequence": state['state']['obs'],
                    "reasoning": ""
                }

                # 处理surrounding_objs
                if 'surrounding_objs' in state['state']:
                    for obj_name, obj_details in state['state']['surrounding_objs'].items():
                        if isinstance(obj_details, list):
                            formatted_state["context"]["surroundings"]["objects"][obj_name] = {
                                "name": obj_name,
                                "desc": obj_details[0] if len(obj_details) > 0 else "",
                                "attribute": obj_details[1:] if len(obj_details) > 1 else []
                            }
                        else:
                            formatted_state["context"]["surroundings"]["objects"][obj_name] = {
                                "name": obj_name,
                                "desc": obj_details.get('desc', ''),
                                "attribute": obj_details.get('attrs', [])
                            }

                # 处理inventory objects
                if state['state']['inv_objs']:
                    for obj_name, obj_details in state['state']['inv_objs'].items():
                        if isinstance(obj_details, list):
                            formatted_state["context"]["inventory"]["objects"][obj_name] = {
                                "name": obj_name,
                                "desc": obj_details[0] if len(obj_details) > 0 else "",
                                "attribute": obj_details[1:] if len(obj_details) > 1 else []
                            }
                        else:
                            formatted_state["context"]["inventory"]["objects"][obj_name] = {
                                "name": obj_name,
                                "desc": obj_details.get('desc', ''),
                                "attribute": obj_details.get('attrs', [])
                            }

                processed_data.append(formatted_state)
            
            except Exception as e:
                print(f"处理游戏序列 {game_idx}, 状态 {state_idx} 时发生错误: {str(e)}")
                print(f"状态数据: {json.dumps(state, ensure_ascii=False, indent=2)}")
    
    print(f"总共跳过了 {skipped_count} 个缺失action的状态")
    
    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

# 使用示例
input_path = '/Users/luao/Downloads/data/train.json'
output_path = '/Users/luao/Downloads/data/processed_train.json'

process_game_data(input_path, output_path)