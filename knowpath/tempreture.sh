#!/bin/bash

# Defines a list containing elements
elements=(0.2 0.4 0.6 0.8 1.0)

# Iterating over a list
for element in "${elements[@]}"; do
    # Run a Python script, passing the element as a parameter
    python main.py -lt deepseek --method knowpath --dataset webqsp --temperature_exploration $element
done
for element in "${elements[@]}"; do
    # Run a Python script, passing the element as a parameter
    python main.py -lt deepseek --method knowpath_wo_sub --dataset webqsp --temperature_exploration $element
done
for element in "${elements[@]}"; do
    # Run a Python script, passing the element as a parameter
    python main.py -lt deepseek --method knowpath_wo_p --dataset webqsp --temperature_exploration $element
done