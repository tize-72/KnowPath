import json
import argparse
from tqdm import tqdm
from utils import *
from prompt_list import *
from alive_progress import alive_bar
import ipdb
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--prompt_methods", type=str,
                        default="cot", help="cot or io.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature", type=int,
                        default=0, help="the temperature")
    parser.add_argument("--result_path", type=str,
                        default="../results/final_sample", help="result saved path.")
    parser.add_argument("--LLM_type", "-lt", type=str,
                        default="qwen2", help="base LLM model.", choices=['qwen2', 'qwen2.5:72b','qwen2.5:32b','qwen2.5','qwq',"qwen2.5:14b",
                                                                          'llama3.1:8b','llama2:13b','llama3',
                                                                          'mistral','mistral-nemo',
                                                                          'glm4',
                                                                          'gpt-3.5-turbo','gpt-4',
                                                                          'phi3:14b',
                                                                          "deepseek",
                                                                          "r1", "deepseek-r1:14b", "deepseek-r1:14b-fp8"
                                                                          ])
    parser.add_argument("--method", type=str,
                        default="cot", help="base LLM model.")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed set.")
    parser.add_argument("--sample_rate", type=int,
                        default=3, help="Sample rate, 10 means 0.1.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="dsd", help="if the M_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    args = parser.parse_args()

    resultpath = f"{args.result_path}/cot"
    os.makedirs(resultpath, exist_ok=True)
    set_seed(args.random_seed)
    with open(f"{args.result_path}/cot/cot_{args.dataset}_{args.LLM_type}.jsonl", 'a+', encoding="UTF-8") as out:
        datas_original, question_string = prepare_dataset(args.dataset)
        datas = get_random_elements(datas_original, args.sample_rate)
        with alive_bar(len(datas),title=args.method,bar='classic') as bar:
            for data in datas:
                if args.prompt_methods == "cot":
                    prompt = cot_prompt + "\n\nQ: " + data[question_string] + "\nA: "
                else:
                    prompt = io_prompt + "\n\nQ: " + data[question_string] + "\nA: "
                results = run_ollama(prompt, args.temperature, args.max_length, args.opeani_api_keys, args.LLM_type)
                out.write(json.dumps({"question": data[question_string], "{}_result".format(args.prompt_methods): results})+'\n')
                
                bar()