import openai
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import re
import time
import pickle
import copy
import json
import requests
from LLM_setup import *
from run_domain import *
from evaluate import *

filepath=r'./records/'
assert os.path.exists(filepath)

def run_domain(domain, model):
    if domain == 'psychology':
        run_psychology(model)
    elif domain == 'cognitive science':
        run_cognitive_science(model)
    elif domain == 'decision making':
        run_decision_making(model)
    elif domain == 'economics':
        run_economics(model)
    elif domain == 'game theory':
        run_game_theory(model)
    elif domain == 'collective rationality':
        run_collective_rationality(model)
    else:
        raise ValueError(f"Invalid domain: {domain}")
    
def evaluate(domain):
    if domain == 'psychology':
        evaluate_psychology()
    elif domain == 'cognitive science':
        evaluate_cognitive_science()
    elif domain == 'decision making':
        evaluate_decision_making()
    elif domain == 'economics':
        evaluate_economics()
    elif domain == 'game theory':
        evaluate_game_theory()
    elif domain == 'collective rationality':
        evaluate_collective_rationality()
    else:
        raise ValueError(f"Invalid domain: {domain}")
    



if __name__ == "__main__":
    domain = 'psychology' # psychology, cognitive science, decision making, economics, game theory, collective rationality
    model = 'gpt-4o'  # You need to first setup LLMs in LLM_setup.py

    models = ['gpt-4o', 'gpt-4', 'gpt-3.5', 'deepseek-v2.5', 'bard', 'text-bison-001', 'text-davinci-003', 'text-davinci-002', 'claude-instant', 'qwen-72b', 'qwen-32b', 'openchat-13b', 'wizardlm-13b', 'vicuna-13b', 'vicuna-7b', 'llama2-13b', 'llama2-7b', 'chatglm2-6b']
    domains = ['psychology', 'cognitive science', 'decision making', 'economics', 'game theory', 'collective rationality']
    for model in models:
        for domain in domains:
            expname = model+'_'+domain
            run_domain(domain, model)

    for domain in domains:
        evaluate(domain)

