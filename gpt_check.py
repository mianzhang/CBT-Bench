

TASK1_CHECK_INTRUCTION = """You are given a question, some choices and an answer text. Your task is to extract the choice from the answer text and reformat it. 
Question: {question}

Choices: {choices}

Answer Text: {answer_text}

Output Format:
Answer: [the letter before the choice]""".strip()


TASK2_CHECK_INTRUCTION = """You are given a question, some choices and an answer text. Your task is to extract the choices from the answer text and reformat them. The question may have multiple choices.
Question: what distortions this patient has?

Choices:
a: all-or-nothing thinking
b: overgeneralization
c: mental filter
d: should statements
e: labeling
f: personalization
g: magnification
h: emotional reasoning
i: mind reading
j: fortune-telling

Answer Text: {answer_text}

Output Format:
Answer: [the letters before the choices seperated by a comma]""".strip()


TASK3_CHECK_INTRUCTION = """You are given a question, some choices and an answer text. Your task is to extract the choices from the answer text and reformat them. The question may have multiple choices.

Question: what core beliefs this patient has?
Choices:
a. helpless
b: unlovable 
c: worthless 

Answer Text: {answer_text}

Output Format:
Answer: [the letters before the choices seperated by a comma]""".strip()


TASK4_CHECK_INTRUCTION = """You are given a question, some choices and an answer text. Your task is to extract the choices from the answer text and reformat them. The question may have multiple choices.

Question: what fine-grained beliefs this patient has?
Choices:
a: I am incompetent
b: I am helpless
c: I am powerless, weak, vulnerable
d: I am a victim
e: I am needy
f: I am trapped
g: I am out of control
h: I am a failure, loser
i: I am defective
j: I am unlovable
k: I am unattractive
l: I am undesirable, unwanted
m: I am bound to be rejected
n: I am bound to be abandoned
o: I am bound to be alone
p: I am worthless, waste
q: I am immoral
r: I am bad - dangerous, toxic, evil
u: I don’t deserve to live

Answer Text: {answer_text}

Output Format:
Answer: [the letters before the choices seperated by a comma]""".strip()




def task1_check_prompts(data, add_system_role=True):
    ret = []

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
        user_query = TASK1_CHECK_INTRUCTION.format(question=query_item['question'], choices=query_choices, answer_text=query_item['generation'])
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
        ret.append(chat)
    return ret


def task234_check_prompts(data, task, add_system_role=True):
    ret = []

    for query_item in data:
        if task == '2': 
            user_query = TASK2_CHECK_INTRUCTION.format(answer_text=query_item['generation'])
        elif task == '3': 
            user_query = TASK3_CHECK_INTRUCTION.format(answer_text=query_item['generation'])
        elif task == '4': 
            user_query = TASK4_CHECK_INTRUCTION.format(answer_text=query_item['generation'])

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
        ret.append(chat)
    return ret


from utils import *
file = 'task4-falcon-7b-chat-shot0-temp0.0.json'
data = load_json(file)
if 'task1' in file:
    conversations = task1_check_prompts(data)
elif 'task2' in file:
    conversations = task234_check_prompts(data, '2')
elif 'task3' in file:
    conversations = task234_check_prompts(data, '3')
elif 'task4' in file:
    conversations = task234_check_prompts(data, '4')
generations = openai_azure_inference(conversations)
choices = []
for gen in generations:
    choices.append(extract_choice(gen).strip())
for i in range(len(data)):
    data[i]['choice'] = choices[i]
dump_json(data, 'checked-' + file)
