import os
from random import Random
import argparse

from transformers import AutoTokenizer
from sklearn.metrics import f1_score

from utils import *
from constants import *


def task4_prompts(data, shot, random_seed=42, return_conversation=False, tokenizer=None, add_system_role=True):
    ret = []
    seed_data = load_json(TASK4_SEED)
    randomizer = Random(random_seed)

    def get_answer_string(item):
        answers = []
        for letter, index in TASK4_INDEX_DICT.items():
            if TASK4_LABELS[index] in item['core_belief_fine_grained']:
                answers.append(letter)
        return ', '.join(answers)
    
    for query_item in data:
        if shot == 0:
            user_query = ZERO_SHOT_TASK4_INTSTRUCTION.format(situation=query_item['situation'], thoughts=query_item['thoughts'])
        else:
            ict_data = randomizer.choices(seed_data, k=shot)
            ict_examples = ""
            for item in ict_data:
                ict_examples += f"Situation: {item['situation']}"
                ict_examples += '\n'
                ict_examples += f"Thoughts: {item['thoughts']}"
                ict_examples += '\n'
                ict_examples += TASK4_CHOICES
                ict_examples += '\n'
                ict_examples += f"Answer: {get_answer_string(item)}"
                ict_examples += '\n\n'
            user_query = K_SHOT_TASK4_INTSTRUCTION.format(
                in_context_examples=ict_examples.strip(),
                situation=query_item['situation'],
                thoughts=query_item['thoughts'])
        if add_system_role:
            chat = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
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
    predictions = []
    follow_format = []
    for gen in generations:
        choices = gen.replace(' ', '').replace('\n', '').split(',')
        pred = [0 for _ in range(len(TASK4_LABELS))]
        for x in choices:
            idx = TASK4_INDEX_DICT.get(x, None)
            if idx is not None:
                pred[idx] = 1
        if any(pred):
            follow_format.append(True)
        else:
            follow_format.append(False)
        predictions.append(pred)
    f1 = f1_score(y_true=answers, y_pred=predictions, average='weighted') 
    return f1, predictions, follow_format


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    args.model_id = HF_MODELS[args.model]

    data = load_json(TASK4_TEST)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if 'gemma' in args.model:
        add_system_role = False
    else:
        add_system_role = True
    if args.model == 'falcon-7b-chat':
        # only support single gpu now
        conversations = task4_prompts(data, args.shot, tokenizer=tokenizer, add_system_role=False, return_conversation=True)
        generations = falcon_mamba_huggingface_inference(
            conversations, args.model_id, args.max_tokens,
            args.temperature, tokenizer)
    else:
        prompts = task4_prompts(data, args.shot, args.random_seed, tokenizer=tokenizer, add_system_role=add_system_role)
        generations = vllm_inference(prompts, args.gpu_num, args.model_id, args.max_tokens, args.temperature, args.stop_strs)
    answers = []
    for item in data:
        ans = [0 for _ in range(len(TASK4_LABELS))]
        for v in item['core_belief_fine_grained']:
            ans[TASK4_LABEL_DICT[v]] = 1
        answers.append(ans)

    f1, predictions, follow_format = check_generation(generations, answers)
    print(f"task 4 weighted F1: {f1:.4f}")
    print(f"{sum(follow_format)}/{len(follow_format)} outputs follow the format")
    for i in range(len(data)):
        data[i]['generation'] = generations[i]
        data[i]['follow_format'] = follow_format[i]
    dump_json(data, f"task4-{args.model}-shot{args.shot}-temp{args.temperature}.json")

