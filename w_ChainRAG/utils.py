import os
import pickle
import faiss
import numpy as np
import json
import logging
from typing import List, Optional, Dict, Any
from string import Template
from LLM import run_ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chain_rag.log", mode="a")
    ]
)
logger = logging.getLogger("chain_rag")

# Dataset configuration
DATASETS = {
    "hotpotqa": {
        "path": "./data/hotpotqa.jsonl",
    },
    "musique": {
        "path": "./data/musique.jsonl",
    },
    "2wikimqa": {
        "path": "./data/2wikimqa.jsonl",
    }
}

def get_embeddings_cache_path(dataset_name: str, text_id: str) -> str:
    """
    Get the path for the embeddings cache file
    
    Args:
        dataset_name: Name of the dataset
        text_id: Text ID
        
    Returns:
        Path to the embeddings cache file (without extension)
    """
    cache_dir = f"cache/{dataset_name}/embeddings"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"embeddings_{text_id}")

def save_embeddings(embeddings: List[np.ndarray], dataset_name: str, text_id: str):
    """
    Save embeddings to cache
    
    Args:
        embeddings: List of embedding vectors
        dataset_name: Name of the dataset
        text_id: Text ID
    """
    embeddings_array = np.array(embeddings).astype('float32')
    
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings_array)
    
    cache_path = get_embeddings_cache_path(dataset_name, text_id)
    faiss.write_index(index, f"{cache_path}.index")
    with open(f"{cache_path}.pkl", 'wb') as f:
        pickle.dump(embeddings_array, f)

def load_embeddings(dataset_name: str, text_id: str) -> Optional[List[np.ndarray]]:
    """
    Load embeddings from cache
    
    Args:
        dataset_name: Name of the dataset
        text_id: Text ID
        
    Returns:
        List of embedding vectors, or None if cache doesn't exist
    """
    cache_path = get_embeddings_cache_path(dataset_name, text_id)
    if os.path.exists(f"{cache_path}.index") and os.path.exists(f"{cache_path}.pkl"):
        try:
            with open(f"{cache_path}.pkl", 'rb') as f:
                embeddings_array = pickle.load(f)
            return list(embeddings_array)
        except Exception as e:
            print(f"Error loading cached embeddings: {str(e)}")
            return None
    return None

def load_dataset(dataset_name: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load the specified dataset
    
    Args:
        dataset_name: Name of the dataset
        num_samples: Number of samples to load, if None load all
        
    Returns:
        List containing the processed data
    """
    if dataset_name not in DATASETS:
        available_datasets = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {available_datasets}")
    
    dataset_path = DATASETS[dataset_name]["path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    logger.info(f"Loading dataset: {dataset_name}")
    
    processed_data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                    
                data = json.loads(line.strip())
                
                # Process different dataset formats
                if dataset_name in ["musique", "2wikimqa"]:
                    processed_item = {
                        'question': data.get('input', ''),
                        'context': data.get('context', ''),
                        'expected_answer': data.get('answers', '')
                    }
                else:
                    # Default format
                    processed_item = {
                        'question': data.get('question', data.get('input', '')),
                        'context': data.get('context', ''),
                        'expected_answer': data.get('answer', data.get('answers', ''))
                    }
                
                processed_data.append(processed_item)
        
        logger.info(f"Successfully loaded {len(processed_data)} samples")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def append_result(result: Dict[str, Any], output_file: str) -> None:
    """
    Append a single result to file
    
    Args:
        result: Single result dictionary
        output_file: Output file path
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False,indent=4)
            f.write("\n")
    except Exception as e:
        print(f"Error appending result: {str(e)}") 


knowpath_prompt = """
You need to answer Question using follow steps:
step1:You need to extract the most relevant topic entities from the Question.\n
step2:Based on the topic entities and Question. List the 15 related knowledge triples from high to low in terms of relevance to the Question . The triples are given in the form of (entity, relation, entity).\n
step3:Based on the knowledge triples you listed, combined with the Question and topic entities, you need to give the final answer. In addition, you need to give the reasoning path. The overall format should be "entity1->relation1->entity2->relation2->entity3->...->end".\n
The answer format is {reasoning_path : ["entity1->relation1->entity2->relation2->entity3->...->end"], "response": "based on the knowledge, the answer to the question $question is xxxx" }\n

Question: $question.\n
Answer:\n

"""
# 定义函数来提取值
def extract_knowledge_content(input_str):
    """
    从字符串中提取knowledge_triples和final_answer的内容
    
    参数:
    input_str: 包含知识内容的字符串
    
    返回:
    tuple: (knowledge_triples, final_answer)
        - knowledge_triples: 知识三元组列表
        - final_answer: 最终答案字符串
    """
    try:
        # 处理空输入
        if not input_str:
            return [], ''
            
        # 找到knowledge_triples的起始和结束位置
        triples_start1 = input_str.find('final answer: {reasoning_path : [') + len('final answer: {reasoning_path : [')
        triples_start2 = input_str.find('Final answer: {\n  "reasoning_path" : [') + len('Final answer: {\n  "reasoning_path" : [')
        triples_start3 = input_str.find('final answer: {\n  "reasoning_path" : [') + len('final answer: {\n  "reasoning_path" : [')
        triples_start = -1
        for item in [triples_start1, triples_start2, triples_start3]:
            if (item > 0) and item > (triples_start):
                triples_start = item

        triples_end1 = input_str.find('], "response"')
        triples_end2 = input_str.find('],\n  "response"')
        triples_end = -1
        for item in [triples_end1, triples_end2]:
            if (item > 0) and item > (triples_end):
                triples_end = item

        # 提取knowledge_triples部分
        triples_str = input_str[triples_start:triples_end]
        # 找到final_answer的起始和结束位置
        answer_start = input_str.find('"response":') + len('"response":')
        answer_end = input_str.rfind('}')
        
        # 提取并清理final_answer
        final_answer = input_str[answer_start:answer_end].strip()
        
        return triples_str, final_answer
        
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return [], ''
    
def knowPath(question,llm):
    """针对单条数据进行knowpath推理

    Args:
        data (_type_): 单条测试数据
        question_string (_type_): 该数据集对应的问题 键名是什么
        args (_type_): 全部参数
    """

    prompt = Template(knowpath_prompt).substitute(question=question)
    response = run_ollama(prompt, '', llm)
    print(response)
    knowledge_triples, final_answer = extract_knowledge_content(response)


    return knowledge_triples, final_answer

def get_context(original_context, knowpath_context):
    final_context = {}
    final_context["important_info"] = original_context
    final_context["additional_info"] = knowpath_context

    return final_context