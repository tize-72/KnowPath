from tqdm import tqdm
import argparse
from utils_knowpath import *
from freebase_func_knowpath import *
import random
from client import *
import ipdb
import time
from alive_progress import alive_bar
import numpy as np
from colorama import Fore, Back, Style, init
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="Choose the dataset.")
    parser.add_argument("--method", type=str,
                        default="knowpath", help="The name of the method.", 
                        choices=['tog', 'base','cot','knowpath','knowpath_wo_p','knowpath_wo_sub'])
    parser.add_argument("--result_path", type=str,
                        default="new_test1", help="Path to save results.")
    parser.add_argument("--max_length", type=int,
                        default=512, help="The max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="Temperature in the exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="Temperature in the reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="The search width for ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="The search depth for ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="Whether to remove unnecessary relations.")
    parser.add_argument("--LLM_type", "-lt", type=str,
                        default="qwen2.5", help="Base LLM model.")
    
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="If the LLM_type is gpt-3.5-turbo or gpt-4, you need to add your OpenAI API keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entity search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="Prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="Random seed.")
    parser.add_argument("--sample_rate", '-sr', type=int,
                        default=0.1, help="Sample rate, 10 means 0.1.")
    parser.add_argument("--max_depth", '-md', type=int,
                        default=3, help="Max depth for exploration.")
    parser.add_argument("--max_entity_width", '-mew', type=int,
                        default=3, help="Max entity width for exploration.")
    parser.add_argument("--is_only_knowpath", '-iok', type=bool,
                        default=False, help="Whether to use KnowPath only for exploration.")
    parser.add_argument("--init_index", '-id', type=int,
                        default=1, help="Initial index for exploration.")
    
    args = parser.parse_args()
    
    # Create result folder
    args.result_path = "../results/"+args.result_path+f"_Depth{str(args.depth)}"
    os.makedirs(args.result_path, exist_ok=True)

    # Set random seed
    set_seed(args.random_seed)
    datas_original, question_string = prepare_dataset(args.dataset)  # Prepare dataset
    # Randomly sample a portion of the data, e.g., 1 means 1% of the data
    datas = get_random_elements(datas_original, args.sample_rate)
    print("Start running KnowPath on %s dataset." % args.dataset)
    
    start_time = time.time()
    with tqdm(total=len(datas)-args.init_index+1, desc=Fore.RED + "KnowPath" + Style.RESET_ALL, colour="green", ncols=150) as pbar:
        for tqdm_index, data in enumerate(datas):
            args.result_dict = get_result_templete()
            tqdm_index = tqdm_index + (args.init_index-1)
            print(f"Current question: {data[question_string]}")
            is_knowpath_answer = False
            if args.method == 'knowpath_wo_sub':
                knowPath(data, question_string, args)

                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / (tqdm_index + 1)) * (100 - tqdm_index - 1) if tqdm_index + 1 > 0 else 0
                formatted_elapsed = format_time(elapsed_time)
                formatted_remaining = format_time(remaining_time)
                pbar.set_postfix({"Cost": formatted_elapsed, "Remain": formatted_remaining})
                pbar.update(1)
            else:
                knowpath_str, response_now, token_num = knowPath(data, question_string, args, True)
                args.result_dict['token_num'] = add_dicts(args.result_dict['token_num'], token_num)
                args.result_dict['call_num'] += 1
                if knowpath_str == '' or knowpath_str == []:
                    knowpath_str = ''
                topic_entity = data['topic_entity']
                
                question = data[question_string]
                args.result_dict["question"] = question
                depth = 0
                while True:
                    flag = False
                    if depth == 0:
                        # First round of exploration
                        result_group = [[] for i in range(len(topic_entity))]
                        for entity_count, entity_id in enumerate(topic_entity):
                            subgraph = SubGraphExploration(entity_id, args)
                            enetity_dict = {entity_id: False}
                            entity_name = [topic_entity[entity_id]]
                            extra_path, new_entity_name, new_enetity_dict = subgraph.subgraph_exploreration_more(enetity_dict,
                                                        entity_name, question, args, depth, '', knowpath_str)
                            result_group[entity_count] = [extra_path, new_entity_name, new_enetity_dict]
                    else:
                        for result_id, result in enumerate(result_group):
                            extra_path, entity_name, enetity_dict = result[0], result[1], result[2]
                            if len(extra_path) == 0:
                                print(f"Subgraph exploration for entity {entity_name} has no new path, stopping this exploration.")
                                continue
                            extra_path, new_entity_name, new_enetity_dict = subgraph.subgraph_exploreration_more(enetity_dict, 
                                                            entity_name, question, args, depth, extra_path, knowpath_str)
                            result_group[entity_count] = ([extra_path, new_entity_name, new_enetity_dict])

                    # After each round of exploration, check if the model can generate an answer from the current subgraph
                    print(f"Inference completed at depth {depth}. Current paths: {[result[0] for result in result_group]}")
                    
                    len_path = 0
                    for result in result_group:
                        len_path += len(result[0])
                    if len_path == 0:
                        print(f"Inference stopped at depth {depth}, no new paths found. Model is providing the answer.")
                        response_now, result_dict  = reasoning_with_knowpath(question, result_group, result_dict, args)
                        result_dict["results"] = response_now
                        result_dict["subgraph"] = result_group
                        save_2_jsonl(result_dict, args)
                        break

                    evaluation_answer, result_dict = evalue_knowpath(result_group, question, args, subgraph.args.result_dict)

                    flag, response = extract_content_from_string(evaluation_answer)

                    subgraph.args.result_dict = copy.deepcopy(result_dict)

                    if depth == args.max_depth:
                        print(f"Inference stopped at depth {depth}, maximum depth reached. Model combines knowledge graph to answer.")
                        result_group = [res[0] for res in result_group]
                        response_now, result_dict  = reasoning_with_knowpath(question, result_group, result_dict, args)
                        result_dict["results"] = response_now
                        result_dict["subgraph"] = result_group
                        result_dict['depth'] = depth
                        save_2_jsonl(result_dict, args)
                        break
                    depth += 1

                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / (tqdm_index + 1)) * (100 - tqdm_index - 1) if tqdm_index + 1 > 0 else 0
                formatted_elapsed = format_time(elapsed_time)
                formatted_remaining = format_time(remaining_time)
                pbar.set_postfix({"Cost": formatted_elapsed, "Remain": formatted_remaining})
                pbar.update(1)