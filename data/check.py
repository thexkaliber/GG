import json

# 读取JSON文件
def read_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 统计空值数量
def count_empty_consequences():
    # 读取数据
    data = read_json_data('processed_train_final.json')
    
    # 初始化计数器
    empty_consequence_count = 0
    total_items = len(data)
    
    # 遍历所有条目
    for item in data:
        # 检查consequence是否为空
        if not item.get('consequence') or item.get('consequence') == '':
            empty_consequence_count += 1
    
    # 打印结果
    print(f"Total items: {total_items}")
    print(f"Empty consequences: {empty_consequence_count}")
    print(f"Empty consequences percentage: {empty_consequence_count/total_items*100:.2f}%")
    print(f"Filled consequences: {total_items - empty_consequence_count}")
    print(f"Filled consequences percentage: {(total_items - empty_consequence_count)/total_items*100:.2f}%")

# 执行统计
count_empty_consequences()
