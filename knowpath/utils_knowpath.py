import os
import json
from string import Template
from knowpath_prompt import *
import time
import openai
import ollama
import ipdb
from openai import OpenAI
import random
import numpy as np
import torch
import re

API_SECRET_KEY = "sk-zk2b28898541776ac946be61e38039b80b3673eb6451f0b9";
BASE_URL = "https://api.zhizengzeng.com/v1/"

Know_API_SECRET_KEY = "sk-xxxx";
Know_BASE_URL = "https://api.xxx.com"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 


def run_ollama(arg, prompt, temperature, max_tokens, openai_api_keys='', engine="qwen2"):
    """Run the large model via Ollama

    Args:
        prompt (type): Prompt text
        temperature (type): Temperature coefficient (default 0.8). Higher values increase response creativity.
        max_tokens (type): Maximum token count for model responses
        openai_api_keys (str, optional): OpenAI API key string. Defaults to ''.
        engine (str, optional): Model type. Defaults to "qwen2".

    Returns:
        type: Returns the large model's inference result
    """
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)

    if 'gpt' in engine:
        client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
        completion = client.chat.completions.create(
        model=engine,# gpt-4o-mini
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        )
        print(completion.choices[0].message.content)
        result = completion.choices[0].message.content
        token_num = {"total": completion.usage.total_tokens, 
                     "input": completion.usage.prompt_tokens, 
                     "output": completion.usage.completion_tokens}
    elif "deep" in engine:
        if arg.method == 'tog':
            client = OpenAI(api_key=ToG_API_SECRET_KEY, base_url=ToG_BASE_URL)
        elif arg.method == 'pog':
            client = OpenAI(api_key=PoG_API_SECRET_KEY, base_url=PoG_BASE_URL)
        elif 'knowpath' in arg.method:
            client = OpenAI(api_key=Know_API_SECRET_KEY, base_url=Know_BASE_URL)
        completion = client.chat.completions.create(
        model="deepseek-chat",# gpt-4o-mini
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        )
        print(completion.choices[0].message.content)
        result = completion.choices[0].message.content
        token_num = {"total": completion.usage.total_tokens, 
                     "input": completion.usage.prompt_tokens, 
                     "output": completion.usage.completion_tokens}
    else:
        response = ollama.chat(model=engine, 

                                messages=messages, 
                                options={
                                "temperature":temperature, 
                                "top_k" : 40, 
                                "top_p" : 0.9, 
                                "num_predict" : max_tokens, 
                                "num_ctx" : 2048 
                                },
                                keep_alive = '10m', 
                                )
        print(response['message']['content'])
        token_num = {"total": response['prompt_eval_count']+response['eval_count'], 
                 "input": response['prompt_eval_count'], 
                 "output": response['eval_count']}
        result = response['message']['content']

    return result, token_num


def save_2_jsonl(result, args):

    """Save results
        Args:
        question (type): description
        answer (type): description
        cluster_chain_of_entities (type): description
        file_name (type): description
        llm_type (type): Which model was used
        method (type): Which method was employed
    """
    result_path_real = f"{args.result_path}/{args.method}"
    os.makedirs(result_path_real, exist_ok=True)
    
    with open(f"{result_path_real}/{args.method}_{args.dataset}_{args.LLM_type}.jsonl", "a") as outfile:
        json_str = json.dumps(result)
        outfile.write(json_str + "\n")


def get_batches(data, batch_size):
    """
    Generates fixed-size sublists (batches) based on the specified batch size.

    Args:
        data (list): Original list data.
        batch_size (int): Size of each batch.

    Yields:
        A sublist of size batch_size per iteration.
        The last batch may be smaller than batch_size.

    Example:
        data = [1, 2, 3, 4, 5, 6]
        batch_size = 2
        Output: Iteratively yields [1, 2], [3, 4], [5, 6]
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def format_time(seconds):
    """Format time as HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def knowPath(data, question_string, args, is_union=False):
    """Perform KnowPath reasoning on a single data instance

    Args:
        data (_type_): Single test data instance
        question_string (_type_): Key name for the corresponding question in the dataset
        args (_type_): Complete parameters
    """
    question = data[question_string]

    prompt = Template(knowpath_prompt).substitute(question=question)
    response, token_num = run_ollama(args, prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    print(response)
    knowledge_triples, final_answer = extract_knowledge_content(response)
    if is_union:
        return knowledge_triples, final_answer, token_num
    args.result_dict["question"] = question
    args.result_dict['results'] = final_answer
    args.result_dict["kg_triples"] = knowledge_triples
    args.result_dict["token_num"] = add_dicts(args.result_dict["token_num"], token_num)
    args.result_dict["call_num"] += 1

    save_2_jsonl(args.result_dict, args) 
    
    return response

def prepare_dataset(dataset_name):
    """Prepare dataset

    Args:
        dataset_name (_type_): Dataset name

    Returns:
        _type_: _description_
    """
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string



def get_random_elements(lst, percentage):
    """
    Randomly selects a specified percentage of elements from a list

    Args:
        lst: Original list
        percentage: Percentage to select (number between 0-100)

    Returns:
        Randomly selected sublist
    """
    num_elements = int(len(lst) * percentage / 100)
    
    return random.sample(lst, num_elements)


def add_dicts(dict1, dict2):
        """Combine keys from both dictionaries

        Args:
            dict1 (_type_): _description_
            dict2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        result = {key: dict1.get(key, 0) + dict2.get(key, 0) for key in dict1}
        
        return result


def evalue_knowpath(kg_subgraph, question, args, dict_in):
    
    prompt = Template(knowpath_evaluation_prompt).substitute(question=question, subgraph=kg_subgraph)

    response, token_num = run_ollama(args, prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    dict_in["call_num"] += 1
    dict_in['token_num'] = add_dicts(dict_in['token_num'], token_num)
    return response, dict_in

def extract_content_from_string(input_str):
    """
    Extracts Answerable and Response values from a string with special characters using string processing methods

    Parameters:
    input_str: String containing JSON content

    Returns:
    tuple: (answerable, response)
    """
    try:
        if not input_str:
            return False, ''
            
        start = input_str.find('{')
        end = input_str.rfind('}')
        
        if start == -1 or end == -1:
            return False, ''
            
        dict_str = input_str[start:end+1]
        
        dict_str = dict_str.replace('\n', '').strip()
        

        answerable_start = dict_str.find('"Answerable":') + len('"Answerable":')
        answerable_end = dict_str.find(',', answerable_start)
        answerable_str = dict_str[answerable_start:answerable_end].strip()
        answerable = True if answerable_str.lower() == 'true' else False

        response_start = dict_str.find('"Response":') + len('"Response":')
        response_end = dict_str.rfind('"')
        response_str = dict_str[response_start:response_end].strip().strip('"')
        
        return answerable, response_str
        
    except Exception as e:
        print(f"Error occurred:{str(e)}")
        return False, ''
    
    

def extract_knowledge_content(input_str):
    """
    Extract knowledge_triples and final_answer content from string

    Args:
        input_str: String containing knowledge content

    Returns:
        tuple: (knowledge_triples, final_answer)
            - knowledge_triples: List of knowledge triples
            - final_answer: Final answer string
    """
    try:

        if not input_str:
            return [], ''
            

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


        triples_str = input_str[triples_start:triples_end]
        answer_start = input_str.find('"response":') + len('"response":')
        answer_end = input_str.rfind('}')
        
        final_answer = input_str[answer_start:answer_end].strip()
        
        return triples_str, final_answer
        
    except Exception as e:
        print(f"Error occurred:{str(e)}")
        return [], ''
    

def reasoning_without_knowpath(question, args):
    prompt = Template(cot_prompt).substitute(question=question)

    response = run_ollama(args, prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return response


def get_result_templete():
    result = {
        "question" : "",
        "results" : "",
        "subgraph" : "",
        "kg_triples" : "",
        "call_num":0,
        "token_num":{"total": 0, 
                 "input": 0, 
                 "output": 0},
        "depth":0,
    }


    return result