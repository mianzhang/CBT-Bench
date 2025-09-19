import json
import argparse


from vllm import LLM, SamplingParams

    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--shot', type=int, default=0, help="The number of in-context examples.")
    parser.add_argument('--stop_strs', type=str, default='')
    parser.add_argument('--random_seed', type=int, default=42)
    return parser


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)


def dump_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)


def vllm_inference(
    prompts,
    gpu_num,
    model_id,
    max_tokens,
    temperature,
    stop_strs='',
):
    if stop_strs == '':
        stop_strs = []
    else:
        stop_strs = stop_strs.split(',')
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, stop=stop_strs)
    llm = LLM(
        model=model_id, dtype='bfloat16', tensor_parallel_size=gpu_num,
        gpu_memory_utilization=0.9)
    outputs = llm.generate(prompts, sampling_params)
    generations = [output.outputs[0].text for output in outputs]
    return generations


def openai_inference(
    messages,
    model,
    end_point,
    api_key,
    api_version,
    temperature,
    max_tokens,
):
    """
    messages = [
        {"role":"system","content":"You are a helpful Assistant."},
        {"role":"user","content":"This is a test"}
    ]
    """
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=end_point,
        api_key=api_key,  
        api_version=api_version
    )
    
    completion = client.chat.completions.create(
        model=model,
        messages = messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    """
    {
        "choices": [
            {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Yes, most of the Azure AI services support customer managed keys. However, not all services support it. You can check the documentation of each service to confirm if customer managed keys are supported.",
                "role": "assistant"
            }
            }
        ],
        "created": 1679001781,
        "id": "chatcmpl-6upLpNYYOx2AhoOYxl9UgJvF4aPpR",
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 39,
            "prompt_tokens": 58,
            "total_tokens": 97
        }
    }
    """ 
    # completion.choices[0].message.content
    return completion


from transformers import FalconMambaForCausalLM
import torch
from tqdm import tqdm
def falcon_mamba_huggingface_inference(
    messages,
    model_id,
    max_new_tokens,
    temperature,
    tokenizer
):
    model = FalconMambaForCausalLM.from_pretrained(model_id).to(f"cuda")
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    ret = []
    for conv in tqdm(messages):
        input_text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        decoded_output = model.generate(input_ids, temperature=temperature, max_new_tokens=max_new_tokens)
        generation = tokenizer.decode(decoded_output[0][len(input_ids[0]):], skip_special_tokens=True)
        # Extract the assistant's response
        # Assume the response starts after the last "Assistant: " in the prompt
        ret.append(generation)
    return ret


import re
def extract_choice(answer_string):
    """
    Extracts the choice from a string formatted as either 'Answer: [choice]' or 'Answer: choice'.

    Args:
        answer_string (str): The string containing the answer.

    Returns:
        str: The extracted choice, or None if no match is found.
    """
    # Define the regex pattern to match both formats
    pattern = r"Answer:\s*(?:\[(.*?)\]|(.*))"
    match = re.search(pattern, answer_string)
    
    # Return the captured group, prioritizing the group inside brackets
    if match:
        return match.group(1) if match.group(1) else match.group(2).strip()
    return None


from openai import AzureOpenAI
from tqdm import tqdm
import os
def openai_azure_inference(conversations, model='gpt-4o', temperature=0.7, max_completion_tokens=2048):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-09-01-preview"
    )
    ret = []
    for conv in tqdm(conversations):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=conv,
                temperature=temperature,
                # max_completion_tokens=max_completion_tokens
            )
            generation = completion.choices[0].message.content
        except:
            generation = '[ERROR]'
        ret.append(generation)
    return ret


