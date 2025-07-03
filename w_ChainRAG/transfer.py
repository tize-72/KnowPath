import json

def json_to_jsonl(input_json_path, output_jsonl_path):
    """
    将JSON文件转换为JSONL文件
    
    参数:
        input_json_path (str): 输入的JSON文件路径
        output_jsonl_path (str): 输出的JSONL文件路径
    """
    try:
        # 读取原始JSON文件
        with open(input_json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        # 写入JSONL文件
        with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
            # 如果原始JSON是数组，则每行写入一个对象
            if isinstance(data, list):
                for item in data:
                    jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')
            # 如果原始JSON是对象，则整个写入一行
            else:
                jsonl_file.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"转换成功！JSONL文件已保存到: {output_jsonl_path}")
    
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # input_path = input("请输入JSON文件路径: ")
    input_path = "/data/chovyzhao/project/ChainRAG/data/musique.json"
    output_path = input_path.rsplit('.', 1)[0] + '.jsonl'  # 自动生成输出路径
    
    json_to_jsonl(input_path, output_path)