from tqdm import tqdm
import argparse
from utils_knowpath import *
from freebase_func_knowpath import *
import random
from client import *
import ipdb
from alive_progress import alive_bar
from concurrent.futures import ThreadPoolExecutor
import concurrent
import numpy as np
from colorama import Fore, Back, Style, init
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--method", type=str,
                        default="knowpath", help="the name of method.", 
                        choices=['tog', 'base','cot','knowpath','knowpath_wo_p','knowpath_wo_sub'])
    parser.add_argument("--result_path", type=str,
                        default="final_sample", help="result saved path.")
    parser.add_argument("--max_length", type=int,
                        default=512, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", "-lt", type=str,
                        default="qwen2", help="base LLM model.", choices=['qwen2', 'qwen2.5:72b','qwen2.5:32b','qwen2.5','qwq',
                                                                          'llama3.1:8b','llama2:13b','llama3',
                                                                          'mistral','mistral-nemo',
                                                                          'glm4',
                                                                          'gpt-3.5-turbo','gpt-4o-mini',
                                                                          'phi3:14b',
                                                                          "deepseek",
                                                                          ])
    
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed set.")
    parser.add_argument("--sample_rate", '-sr',type=int,
                        default=1, help="Sample rate, 10 means 0.1.")
    parser.add_argument("--max_depth",'-md', type=int,
                        default=3, help="Exploration max depth.")
    parser.add_argument("--max_entity_width",'-mew', type=int,
                        default=3, help="Exploration max_entity_width.")
    parser.add_argument("--is_only_knowpath",'-iok', type=bool,
                        default=False, help="Wether only use knowpath to explore.")
    parser.add_argument("--init_index",'-id', type=int,
                        default=1, help="Exploration max depth.")
    
    args = parser.parse_args()
    # Create results folder
    args.result_path = "../results/"+args.result_path+'test'
    os.makedirs(args.result_path, exist_ok=True)


    # Set random seed
    set_seed(args.random_seed)
    datas_original, question_string = prepare_dataset(args.dataset) # Test data
    # Randomly sample data from the dataset for testing. 1 means 1%
    datas = get_random_elements(datas_original, args.sample_rate)
    print("Start Running knowpath on %s dataset." % args.dataset)
    
    start_time = time.time()
    with tqdm(total=len(datas)-args.init_index+1,desc=Fore.RED + "KnowPath"+ Style.RESET_ALL, colour="green",ncols=150) as pbar:
    # with alive_bar(len(datas),title=args.method,bar='classic') as bar:
        for tqdm_index, data in enumerate(datas):
            args.result_dict = get_result_templete()
            tqdm_index = tqdm_index + (args.init_index-1)
            print(f"The current issue is {data[question_string]}")
            is_knowpath_answer = False
            if args.method == 'knowpath_wo_sub':
                knowPath(data, question_string, args)

                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / (tqdm_index + 1)) * (100 - \
                                                                        tqdm_index - 1) if tqdm_index + 1 > 0 else 0
                formatted_elapsed = format_time(elapsed_time)# Format time
                formatted_remaining = format_time(remaining_time)
                pbar.set_postfix({"Cost": formatted_elapsed, "Remain": formatted_remaining})# Dynamically update right-side information
                pbar.update(1)
            else:
                try:
                    knowpath_str, response_now, token_num = knowPath(data, question_string, args, True)
                    args.result_dict['token_num'] = add_dicts(args.result_dict['token_num'], token_num)
                    args.result_dict['call_num'] += 1
                    if knowpath_str == '' or knowpath_str == []:
                        knowpath_str = ''
                    topic_entity = data['topic_entity']
                    
                    # 1.Generate LLM subgraph subgraph 2.Explore KG subgraph 3.Integrate answers
                    question = data[question_string]
                    args.result_dict["question"] = question
                    for depth in range(args.max_depth):
                        flag = False
                        if depth == 0:
                            # First round exploration
                            result_group = [[] for i in range(len(topic_entity))]
                            for entity_count, entity_id in enumerate(topic_entity):
                                # For each entity, perform subgraph expansion.
                                subgraph = SubGraphExploration(entity_id, args)
                                enetity_dict = {entity_id : False}
                                entity_name = [topic_entity[entity_id]]
                                extra_path, new_entity_name, new_enetity_dict = subgraph.subgraph_exploreration_more(enetity_dict, 
                                                            entity_name, question, args, depth, '', knowpath_str)
                                result_group[entity_count] = [extra_path, new_entity_name, new_enetity_dict]
                        else:
                            for result_id, result in enumerate(result_group):
                                extra_path, entity_name, enetity_dict = result[0], result[1], result[2]
                                if len(extra_path) == 0:
                                    print(f"The previous exploration on this topic entity's path returned empty results. Terminate this path and only explore the remaining path.")
                                    continue
                                extra_path, new_entity_name, new_enetity_dict = subgraph.subgraph_exploreration_more(enetity_dict, 
                                                                entity_name,question,args, depth, extra_path, knowpath_str)
                                result_group[entity_count] = ([extra_path, new_entity_name, new_enetity_dict])
                        print(f"Depth {depth} reasoning completed. Current overall reasoning path:{[result[0] for result in result_group]}")
                        len_path = 0
                        for result in result_group:
                            len_path += len(result[0])
                        if len_path== 0:
                            print(f"Reasoning stopped at round {depth}. No new knowledge acquired. Model provided response.")
                            result_dict["results"] = response_now
                            save_2_jsonl(result_dict, args)
                            break
                        evaluation_answer, result_dict = evalue_knowpath(result_group, question, args, subgraph.args.result_dict)

                        flag, response = extract_content_from_string(evaluation_answer)
                        if flag:
                            # If True, provide the answer given by the large model.
                            print(f"KnowPath obtained the answer at round {depth}, terminating this exploration.")
                            args.result_dict = subgraph.args.result_dict
                            args.result_dict["question"] = question
                            args.result_dict["depth"] = depth
                            args.result_dict["results"] = response
                            args.result_dict["subgraph"] = result_group
                            save_2_jsonl(args.result_dict, args)
                            break
                        else:
                            # Indicates the model didn't obtain an answer this round. Continue exploration.
                            print(f"The large model couldn't obtain an answer in round {depth}. Continuing exploration.")
                        subgraph.args.result_dict = copy.deepcopy(result_dict)

                        if depth == args.max_depth-1:
                            print(f"Reasoning stopped at round {depth} (maximum depth reached). Model generated response.")
                            result_dict["results"] = response_now
                            save_2_jsonl(result_dict, args)

                    elapsed_time = time.time() - start_time
                    remaining_time = (elapsed_time / (tqdm_index + 1)) * (100 - \
                                                                            tqdm_index - 1) if tqdm_index + 1 > 0 else 0
                    formatted_elapsed = format_time(elapsed_time)# Format time
                    formatted_remaining = format_time(remaining_time)
                    pbar.set_postfix({"Cost": formatted_elapsed, "Remain": formatted_remaining})# Dynamically update right-side information
                    pbar.update(1)
                except:
                    print(f"Reasoning failed at round {depth}. Model provided response.")
                    response = reasoning_without_knowpath(question, args)
                    args.result_dict["results"] = response_now
                    save_2_jsonl(args.result_dict, args)

                    elapsed_time = time.time() - start_time
                    remaining_time = (elapsed_time / (tqdm_index + 1)) * (100 - \
                                                                            tqdm_index - 1) if tqdm_index + 1 > 0 else 0
                    formatted_elapsed = format_time(elapsed_time)# Format time
                    formatted_remaining = format_time(remaining_time)
                    pbar.set_postfix({"Cost": formatted_elapsed, "Remain": formatted_remaining})# Dynamically update right-side information
                    pbar.update(1)

                    continue