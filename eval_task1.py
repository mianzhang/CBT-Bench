import os
from random import Random

from transformers import AutoTokenizer
from utils import *
from constants import *


def task1_prompts(data, shot, random_seed, return_conversation=False, tokenizer=None, add_system_role=True):
    ret = []
    seed_data = load_json(TASK1_SEED)
    randomizer = Random(random_seed)
    
    def get_choices(item):
        choices = ''
        for key in ['a', 'b', 'c', 'd', 'e']:
            ans = item.get(key, None)
            if not ans:
                continue
            choices += f"{key}: {ans}\n"
        return choices.strip()

    for query_item in data:
        query_choices = get_choices(query_item)
        if shot == 0:
            user_query = ZERO_SHOT_TASK1_INTSTRUCTION.format(question=query_item['question'], choices=query_choices)
        else:
            ict_data = randomizer.choices(seed_data, k=shot)
            ict_examples = ""
            for item in ict_data:
                ict_choices = get_choices(item) 
                ict_examples += item['question']
                ict_examples += '\n'
                ict_examples += ict_choices
                ict_examples += '\n'
                ict_examples += f"Answer: {item['answer'][0]}"
                ict_examples += '\n\n'
            user_query = K_SHOT_TASK1_INTSTRUCTION.format(
                in_context_examples=ict_examples.strip(),
                question=query_item['question'],
                choices=query_choices)
        if add_system_role:
            chat = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        else:
            chat = [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        if return_conversation:
            ret.append(chat)
        else:
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            ret.append(prompt)
    return ret


def check_generation(generations, answers):
    judges = []
    for gen, ans in zip(generations, answers):
        pred = gen.strip()[0]
        judges.append(pred == ans)
    return judges


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    args.model_id = HF_MODELS[args.model]

    data = load_json(TASK1_TEST)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if 'gemma' in args.model:
        add_system_role = False
    else:
        add_system_role = True
    if args.model == 'falcon-7b-chat':
        # only support single gpu now
        conversations = task1_prompts(data, args.shot, args.random_seed, tokenizer=tokenizer, add_system_role=False, return_conversation=True)
        generations = falcon_mamba_huggingface_inference(
            conversations, args.model_id, args.max_tokens,
            args.temperature, tokenizer)
    else:
        prompts = task1_prompts(data, args.shot, args.random_seed, tokenizer=tokenizer, add_system_role=add_system_role)
        generations = vllm_inference(
            prompts, args.gpu_num, args.model_id, args.max_tokens,
            args.temperature, args.stop_strs)
    answers = [item['answer'][0] for item in data]

    judges = check_generation(generations, answers)
    acc = sum(judges) / len(data)
    print(f"task 1 acc: {sum(judges)}/{len(data)} = {acc:.4f}")

    for i in range(len(data)):
        data[i]['generation'] = generations[i]
        data[i]['judge'] = judges[i]
    dump_json(data, f"task1-{args.model}-shot{args.shot}-temp{args.temperature}.json")
