import argparse
from utils import *
import sys;sys.path.append('/data/zhaoqi/project/knowpath')
import os
import ipdb

def remove_yes_no_brackets_simple(text):
    replacements = ['{yes}', '{Yes}', '{no}', '{No}']
    for item in replacements:
        text = text.replace(item, '')
    return text


def write_wrong_list(data_list, file):
    with open(file, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

def extract_final_answer(text):
    try:
        start_index = text.find('"final answer": "') + len('"final answer": "')
        if start_index == -1:
            return None
            
        end_index = text.find('"]', start_index)
        if end_index == -1:
            return None

        final_answer = text[start_index:end_index]
        return final_answer
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def return_right_list(wrong_list, idx, data, answers, response):
    right_dict = {
            "id" : idx,
            "question" : data['question'], 
            "true" : answers,
            "answer" : response
        }
    wrong_list.append(right_dict)

    return wrong_list


def check_strings_contain(str1, str2):
    return str1 in str2 or str2 in str1 or str1==str2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--method", type=str,
                        default="knowpath", help="choose the dataset.")
    parser.add_argument("--llm_type", '-lt', type=str,
                        default="qwen2", help="llm model type")
    parser.add_argument("--output_file", type=str,
                        default="x", help="the output file name.")
    parser.add_argument("--constraints_refuse", type=bool,
                        default=False, help="LLM may have refuse erorr, enable this option to skip current sample.")
    parser.add_argument("--result_path", type=str,
                        default="final_sample", help="result saved path.")
    args = parser.parse_args()


    args.output_file = f'../results/{args.result_path}/{args.method}/{args.method}_{args.dataset}_{args.llm_type}.jsonl'
    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, args.output_file)

    num_right = 0
    num_error = 0
    wrong_list = []
    for idx, data in enumerate(output_datas):
        answers = align(args.dataset, question_string, data, ground_truth_datas)
        if args.method == 'cot':
            result = extract_final_answer(data["cot_result"])
        else:
            result = extract_final_answer(data["results"])
        if result is None: 
            if args.method == 'cot':
                result=data["cot_result"]
            else:
                result=data["results"]
        if exact_match(result, answers):
            num_right += 1
            wrong_list = return_right_list(wrong_list, idx, data, answers, result)
        else:
            num_error += 1


    print("Exact Match: {}".format(float(num_right/len(output_datas))))
    print("right: {}, error: {}".format(num_right, num_error))
    f1 = float(num_right/len(output_datas))
    input = {"method":args.method, 
             "llm_type":args.llm_type, 
             "f1":round(f1, 4), 
             "number":f'{num_right}/{num_error}/{len(output_datas)}'}
    with open('eval.json', 'a') as f:
        json.dump(input, f)
        f.write('\n')
    wrong_folder = f'../results/final_sample/{args.method}/wrong'
    os.makedirs(wrong_folder, exist_ok=True)
    args.wrong_path = f'{wrong_folder}/{args.method}_{args.dataset}_{args.llm_type}.json'
    write_wrong_list(wrong_list, args.wrong_path)