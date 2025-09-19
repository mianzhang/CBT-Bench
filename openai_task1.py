
import os
import argparse
import time

from tqdm import tqdm

from utils import *
from constants import *
from eval_task1 import task1_prompts


def check_generation(generations, answers):
    judges = []
    for gen, ans in zip(generations, answers):
        pred = gen[0]
        judges.append(pred == ans)
    return judges


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    end_point = os.environ['END_POINT']
    api_key = os.environ['API_KEY']
    api_version = os.environ['API_VERSION']
    RATE_LIMIT = 1  # 1 request per second

    path = TASK1_TEST
    data = load_json(path)
    conversations = task1_prompts(data, args.shot, args.random_seed, return_conversation=True)
    generations = []
    for conv in tqdm(conversations):
        completion = openai_inference(conv, args.model, end_point, api_key, api_version, args.temperature, args.max_tokens)
        generations.append(completion.choices[0].message.content)
        # Wait to respect the rate limit
        time.sleep(1 / RATE_LIMIT)  # Sleep for the time needed to maintain the rate limit
        
    answers = [item['answer'][0] for item in data]

    judges = check_generation(generations, answers)
    acc = sum(judges) / len(data)
    print(f"task 1 acc: {sum(judges)}/{len(data)} = {acc:.4f}")


    for i in range(len(data)):
        data[i]['generation'] = generations[i]
        data[i]['judge'] = judges[i]
    dump_json(data, f"task1-{args.model}-shot{args.shot}-temp{args.temperature}.json")
