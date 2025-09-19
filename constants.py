import os

### Data
CBT_DATA_DIR = ''
TASK1_TEST = os.path.join(CBT_DATA_DIR, 'qa_test.json')
TASK1_SEED = os.path.join(CBT_DATA_DIR, 'qa_seed.json')
TASK2_TEST = os.path.join(CBT_DATA_DIR, 'distortions_test.json')
TASK2_SEED = os.path.join(CBT_DATA_DIR, 'distortions_seed.json')
TASK3_TEST = os.path.join(CBT_DATA_DIR, 'core_major_test.json')
TASK3_SEED = os.path.join(CBT_DATA_DIR, 'core_major_seed.json')
TASK4_TEST = os.path.join(CBT_DATA_DIR, 'core_fine_test.json')
TASK4_SEED = os.path.join(CBT_DATA_DIR, 'core_fine_seed.json')


### Model
HF_CACHE_DIR = ''
HF_MODELS = {
    'llama3.2-1b-chat': os.path.join(HF_CACHE_DIR, 'meta-llama/Llama-3.2-1B-Instruct'),
    'llama3.2-3b-chat': os.path.join(HF_CACHE_DIR, 'meta-llama/Llama-3.2-3B-Instruct'),
    'llama3.1-8b-chat': os.path.join(HF_CACHE_DIR, 'meta-llama/Meta-Llama-3.1-8B-Instruct'),
    'llama3.1-70b-chat': os.path.join(HF_CACHE_DIR, 'meta-llama/Meta-Llama-3.1-70B-Instruct'),
    'mistral-7b-chat': os.path.join(HF_CACHE_DIR, 'mistralai/Mistral-7B-Instruct-v0.3'),
    'gemma2-9b-it': os.path.join(HF_CACHE_DIR, 'google/gemma-2-9b-it'),
    'mixtral-7-8B-chat': os.path.join(HF_CACHE_DIR, 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
    'mixtral-7-8B': os.path.join(HF_CACHE_DIR, 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
    'falcon-7b-chat': os.path.join(HF_CACHE_DIR, 'tiiuae/falcon-mamba-7b-instruct'),
}


MISTRAL_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


### Task 1 prompts
ZERO_SHOT_TASK1_INTSTRUCTION = """You are taking a CBT exam and doing multiple choices questions. Each question has only one right choice.
{question}
{choices}
Please only output the letter corresponding to the choice.""".strip()


K_SHOT_TASK1_INTSTRUCTION = """You are taking a CBT exam and doing multiple choices questions. Each question has only one right choice.
{in_context_examples}
{question}
{choices}
Answer:
Please only output the letter corresponding to the choice.""".strip()


### Task 2 prompts
TASK2_LABELS = ["all-or-nothing thinking", "overgeneralization","mental filter","should statements","labeling","personalization","magnification","emotional reasoning","mind reading","fortune-telling"]
TASK2_LABEL_DICT = dict((v, k) for k, v in enumerate(TASK2_LABELS))
TASK2_INDEX_DICT = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9}

TASK2_CHOICES = """Choices:
a: all-or-nothing thinking
b: overgeneralization
c: mental filter
d: should statements
e: labeling
f: personalization
g: magnification
h: emotional reasoning
i: mind reading
j: fortune-telling""".strip()

ZERO_SHOT_TASK2_INTSTRUCTION = """You are a CBT therapist and now need to determine the cognitive distortions of a patient from his current situation and thoughts. Each patient may have **up to 3** distortions.
Situation: {situation}
Thoughts: {thoughts}
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
Answer:
Please only output the letters corresponding to the choices. Multiple choices should be separated by a comma.""".strip()


K_SHOT_TASK2_INTSTRUCTION = """You are a CBT therapist and now need to determine the cognitive distortions of a patient from his current situation and thoughts. Each patient may have **up to 3** distortions. 
{in_context_examples}
Situation: {situation}
Thoughts: {thoughts}
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
Answer:
Please only output the letters corresponding to the choices. Multiple choices should be separated by a comma.""".strip()


### Task 3 prompts
TASK3_LABELS = ["helpless", "unlovable","worthless"]
TASK3_LABEL_DICT = dict((v, k) for k, v in enumerate(TASK3_LABELS))
TASK3_INDEX_DICT = {'a': 0, 'b': 1, 'c': 2}
TASK3_CHOICES = """Choices:
a: helpless
b: unlovable 
c: worthless""".strip()


ZERO_SHOT_TASK3_INTSTRUCTION = """You are a CBT therapist and now need to determine the major core beliefs of a patient from his current situation and thoughts. Each patient may have multiple core beliefs.
Situation: {situation}
Thoughts: {thoughts}
Question: what core beliefs this patient has?
a: helpless
b: unlovable 
c: worthless 
Answer:
Please only output the letters corresponding to the choices. Multiple choices should be separated by a comma.""".strip()


K_SHOT_TASK3_INTSTRUCTION = """You are a CBT therapist and now need to determine the major core beliefs of a patient from his current situation and thoughts. Each patient may have multiple core beliefs.
{in_context_examples}
Situation: {situation}
Thoughts: {thoughts}
Question: what core beliefs this patient has?
Choices:
a. helpless
b: unlovable 
c: worthless 
Answer:
Please only output the letters corresponding to the choices. Multiple choices should be separated by a comma.""".strip()


### Task 4 prompts
TASK4_LABELS = ["I am incompetent","I am helpless","I am powerless, weak, vulnerable","I am a victim","I am needy","I am trapped","I am out of control","I am a failure, loser","I am defective","I am unlovable","I am unattractive","I am undesirable, unwanted","I am bound to be rejected","I am bound to be abandoned","I am bound to be alone","I am worthless, waste","I am immoral","I am bad - dangerous, toxic, evil","I don’t deserve to live"]
TASK4_LABEL_DICT = dict((v, k) for k, v in enumerate(TASK4_LABELS))
TASK4_INDEX_DICT = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 'u': 18}
TASK4_CHOICES = """Choices:
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
u: I don’t deserve to live""".strip()


ZERO_SHOT_TASK4_INTSTRUCTION = """You are a CBT therapist and now need to determine the fine-grained beliefs of a patient from his current situation and thoughts. Each patient may have **up to 9** fine-grained beliefs. Now answer the following question:
Situation: {situation}
Thoughts: {thoughts}
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
Answer:
Please only output the letters corresponding to the choices. Multiple choices should be separated by a comma.""".strip()


K_SHOT_TASK4_INTSTRUCTION = """You are a CBT therapist and now need to determine the fine-grained beliefs of a patient from his current situation and thoughts. Each patient may have **up to 9** fine-grained beliefs.
{in_context_examples}
Situation: {situation}
Thoughts: {thoughts}
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
Answer:
Please only output the letters corresponding to the choices. Multiple choices should be separated by a comma.""".strip()
