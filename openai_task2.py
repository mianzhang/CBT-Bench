
import os
import time

from tqdm import tqdm

from utils import *
from constants import *
from eval_task2 import check_generation, task2_prompts


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    end_point = os.environ['END_POINT']
    api_key = os.environ['API_KEY']
    api_version = os.environ['API_VERSION']
    RATE_LIMIT = 1  # 1 request per second

    path = TASK2_TEST
    data = load_json(path)
    conversations = task2_prompts(data, args.shot, return_conversation=True)
    generations = []
    for conv in tqdm(conversations):
        try:
            completion = openai_inference(conv, args.model, end_point, api_key, api_version, args.temperature, args.max_tokens)
            generations.append(completion.choices[0].message.content)
        except:
            generations.append("[ERROR]")
        # Wait to respect the rate limit
        time.sleep(1 / RATE_LIMIT)  # Sleep for the time needed to maintain the rate limit
    answers = []
    for item in data:
        ans = [0 for _ in range(len(TASK2_LABELS))]
        for v in item['distortions']:
            ans[TASK2_LABEL_DICT[v]] = 1
        answers.append(ans)

    f1, predictions, follow_format = check_generation(generations, answers)
    print(f"task 2 weighted F1: {f1:.4f}")
    print(f"{sum(follow_format)}/{len(follow_format)} outputs follow the format")
    for i in range(len(data)):
        data[i]['generation'] = generations[i]
        data[i]['follow_format'] = follow_format[i]
    if args.shot:
        dump_json(data, f"task2-{args.model}-shot{args.shot}-temp{args.temperature}-seedidx{args.seed_idx}.json")
    else:
        dump_json(data, f"task2-{args.model}-shot{args.shot}-temp{args.temperature}.json")