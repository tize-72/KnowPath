import openai
import time
import json
import ollama
import random
import numpy as np
import torch
import openai
from openai import OpenAI

API_SECRET_KEY = "sk-xxx";
BASE_URL = "https://api.xxx.com/v1/"
DEEP_API_SECRET_KEY = "sk-xxx";
DEEP_BASE_URL = "https://api.xxx.com"
def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

def get_random_elements(lst, percentage):

    num_elements = int(len(lst) * percentage / 100)
    
    return random.sample(lst, num_elements)


def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    if "llama" not in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"  # your local llama server port
        engine = openai.Model.list()["data"][0]["id"]
    else:
        openai.api_key = opeani_api_keys

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    print("start openai")
    while(f == 0):
        try:
            response = openai.ChatCompletion.create(
                    model=engine,
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)
            result = response["choices"][0]['message']['content']
            f = 1
        except:
            print("openai error, retry")
            time.sleep(2)
    print("end openai")
    return result

def run_ollama(prompt, temperature, max_tokens, openai_api_keys='', engine="qwen2"):
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

    if 'gpt' not in engine:
        response = ollama.chat(model=engine, 

                                messages=messages, 
                                options={
                                "temperature":temperature, 
                                "top_k" : 40, 
                                "top_p" : 0.9, 
                                "num_predict" : max_tokens, 
                                "num_ctx" : 2048 
                                },
                                keep_alive = '1m', 
                                )
        print(response['message']['content'])
        result = response['message']['content']
    else:
        client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
        # model="gpt-3.5-turbo",# gpt-4o-mini
        model=engine,# gpt-4o-mini
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        )
        print(resp.choices[0].message.content)
        result = resp.choices[0].message.content

    return result


def prepare_dataset(dataset_name):
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
        print("dataset not found")
        exit(-1)
    return datas, question_string