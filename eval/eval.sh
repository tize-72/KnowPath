#!/bin/bash   # Make sure to use bash to execute

# Check if eval.json exists, if so, clear it
if [ -f "eval.json" ]; then
    echo "Found eval.json and is clearing it..."
    : > eval.json  # Clear file contents
else
    echo "eval.json does not exist and will be created when executing the Python script"
fi

echo "Starting Python script..."

# Define the function
process_item() {
    local llm=$1
    local method=$2
    echo "LLM: $llm with Method: $method"
    python knowpath_eval.py --method $method -lt $llm
}
# Define the function base
process_item_base() {
    local llm=$1
    echo "LLM: $llm with Method: base"
    python knowpath_eval.py -lt $llm --method base
}

# Define the function cot
process_item_cot() {
    local llm=$1
    echo "LLM: $llm with Method: cot"
    python knowpath_eval.py -lt $llm --method cot
}

# Define the function knowpath
process_item_knowpath() {
    local llm=$1
    echo "LLM: $llm with Method: knnowpath"
    python eval.py -lt $llm --method knowpath

}

# Define the function knowpath_wo_sub
process_item_knowpath_wo_sub() {
    local llm=$1
    echo "LLM: $llm with Method: knowpath_wo_sub"
    python knowpath_eval.py -lt $llm --method knowpath_wo_sub

}

# Define the function knowpath_wo_p
process_item_knowpath_wo_p() {
    local llm=$1
    echo "LLM: $llm with Method: knowpath_wo_p"
    python knowpath_eval.py -lt $llm --method knowpath_wo_p

}

llm_types=("qwen2")
# llm_types=("gpt-3.5-turbo" "deepseek")


# Nested for loop base
for llm in "${llm_types[@]}"; do
    process_item_base "$llm"
done
# Nested for loops cot
for llm in "${llm_types[@]}"; do
    process_item_cot "$llm"
done
# Nested for loop knowpath
for llm in "${llm_types[@]}"; do
    process_item_knowpath "$llm"
done
# Nested for loop knowpath_wo_sub
for llm in "${llm_types[@]}"; do
    process_item_knowpath_wo_sub "$llm"
done
# Nested for loop knowpath_wo_p
for llm in "${llm_types[@]}"; do
    process_item_knowpath_wo_p "$llm"
done
# You can continue to add more Python scripts

# Check if eval.json exists and display the content
if [ -f "eval.json" ]; then
    echo -e "The content of \neval.json is as follows:"
    cat eval.json
else
    echo "Warning: eval.json file not found after execution completed"
fi
