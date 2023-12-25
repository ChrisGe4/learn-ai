# Why finetune

## What is finetuning

Taking these general purpose models like GPT-3 and specializing them into something 
like ChatGPT, the specific chat use case to make it chat well, or using GPT-4 
and turning that into a specialized GitHub co-pilot use case to auto-complete code. 

## What does finetuning do for the model

- Makes it possible for you to give it a lot more data than what fits into 
the prompt so that your model can learn from that data rather than just get access to it

e.g. A model that is fine-tuned on dermatology data however might take in the same symptoms and be 
able to give you a much clearer, more specific diagnosis. 

- Helps steer the model to more consistent outputs or more consistent behavior. 

For example, you can see the base model here. When you ask it, what's your first name? 
It might respond with, what's your last name? Because it's seen so much survey data out there of different questions. 
 
So it doesn't even know that it's supposed to answer that question. 
But a fine-tuned model by contrast, when you ask it, what's your 
first name? would be able to respond clearly. My first name is Sharon. 

- Reduces hallucinations
- Customizes the model to a specific use case
- Process is similar to the model's earlier training

## Prompt Engineering vs. Finetuning

![dia](doc-data/prompt-vs-finetuning.png)

### Prompt Pros Explain

- smaller upfront cost, so you don't really need to think about cost, since every single time you ping 
the model, it's not that expensive.
-  RAG, to connect more of your data to it, to selectively choose what kind of data goes into the prompt.
 
### Prompt Cons Explain

- Oftentimes when you do try to fit in a ton of data, unfortunately it will forget a lot of that data. 
- Hallucination, which is when the model does make stuff up and it's hard to **correct** that incorrect information that it's already learned.
- RAG often miss the right data,get the incorrect data and cause the model, to output the wrong thing.

### Finetune Pros Explain

- you can correct that incorrect information that it may have learned before, or even put in recent information that it hadn't learned about previously
- There's less compute cost afterwards if you do fine-tune a smaller model and this is particularly relevant if you expect to hit the model a lot of times
- RAG connects it with far more data as well even after it's learned all this information


## Benefits of finetuning your own LLM

- Performance
  - stops the LLM from making stuff up, especially around your domain.
  - It can be far more consistent.
  - better at moderating. Reduce unwanted info, esp. about your company

- Privacy
  - in your VPC or on premise
  - prevents data leakage and data breaches that might happen on off the shelf, third party solutions.
    This is one way to keep that data safe that you've been collecting for a while.

- Cost
  - cost transparency.
  - lower the cost per request. fine tuning a smaller LLM can actually help you do that. 
  - greater control over costs and a couple other factors as well. 

- Reliability
  - control uptime
  - lower latency. You can greatly reduce the latency for certain applications like autocomplete.
  - moderation. Basically, if you want the model to say, I'm sorry to certain things, or to say, I don't know
    to certain things, or even to have a custom response, This is one way to actually provide those guardrails to the model. 

# Where finetuning fits in

## Finetuning data: compare to pretraining and basic preparation

### Pretraining

First step before fine-tuning even happens.

Model at the start
  - zero knowledge about the world: weights are completely random
  - can't form english words: no language skill
- Next token prediction is the objective
- Giant corpus of text data that often scraped from the internet: "unlabeled". A lot of manual work to 
  getting this data set to be effective for model pre-training even after many cleaning processes.
- often called self-supervised learning because the model is essentially supervising itself with next token prediction.

After training
- Learns language
- Learns knowledge

It is just trying to predict the next token and it's reading the entire Internet's worth of data to do so. 

## What is "data" scraped from the internet

- However the data set of the closed source models from large companies are not very public.
- Open source effort by EleutherAI to create a dataset called The Pile.
- pre-training step is pretty expensive and time-consuming because it's so time-consuming to have the model go through all of this data,
  go from absolutely randomness to understanding some of these texts

## Finetuning after pretraining

pre-training is really that first step that gets you that base model. It's not useful at this moment.

you can use fine-tuning to get a fine-tuned model. And actually, even a fine-tuned model, you can continue adding fine-tuning steps afterwards. 
So fine-tuning really is a step afterwards. 

### Finetuning usually refers to training further

- Can also be self-supervised unlabeled data
- Can be "labeled" data you curated to make it much more structured for the model 
to learn about
- same training obj: next token prediction

**one thing that's key that differentiates fine-tuning from pre-training is that there's much less data needed.**

Note: Definition of fine-tuning here is updating the weights of the entire model, not just part of it(vs. fine-tuning on ImageNet).

In summary, all we're doing is changing up the data so that it's more structured in a way, and the model can be more consistent in 
outputting and mimicking that structure.


## What is finetuning doing for you?

behavior change - both - gain knowledge 

- one giant category is just behavior change. You're changing the behavior of the model.
  - learn to respond much more consistently
  - learn to focus, e.g. moderation
  - teasing out capability, e.g. better at conversation. before we would have to do a lot of prompt engineering in 
    order to tease that information out.
- gain knowledge
  - Increase knowledge of new specific topics that are not in that base pre-trained model
  - Correct old incorrect information 

## Tasks to finetune

- just text in, text out for LLMs
   - extracting text: put text in and you get less text out.
      - reading:  extracting keywords, topics
      - route the chat, for example, to some API or otherwise
      - agent capabilities: planning, reasoning, self-critic, tool use, etc.
   - expansion: you put text in, and you get more text out
      - writing: chatting, writing emails/code
 - task clarity is key indicator of success: clarity really means knowing what good output looks like, what bad output looks like, but also what better output looks like.


## Practice: first time finetuning 

1. identify tasks by just prompt engineering a large LLM and that could be chat GPT.
2. find a task you see on LLM doing ok at.
3. pick one task.
4. get ~1000(golden number) input and outputs for that task. make sure that these inputs and outputs are better than the okay result from that LLM before.
5. finetune a small LLM on this data.


## Sample code

Various ways of formatting your data

```py
filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
instruction_dataset_df

examples = instruction_dataset_df.to_dict()
text = examples["question"][0] + examples["answer"][0]
text

if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]

prompt_template_qa = """### Question:
{question}

### Answer:
{answer}"""

question = examples["question"][0]
answer = examples["answer"][0]

text_with_prompt_template = prompt_template_qa.format(question=question, answer=answer)
text_with_prompt_template

prompt_template_q = """### Question:
{question}

### Answer:"""


num_examples = len(examples["question"])
finetuning_dataset_text_only = []
finetuning_dataset_question_answer = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]

  text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
  finetuning_dataset_text_only.append({"text": text_with_prompt_template_qa})

  text_with_prompt_template_q = prompt_template_q.format(question=question)
  finetuning_dataset_question_answer.append({"question": text_with_prompt_template_q, "answer": answer})


pprint(finetuning_dataset_text_only[0])
pprint(finetuning_dataset_question_answer[0])
```
 
Common ways of storing your data

```py
with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
    writer.write_all(finetuning_dataset_question_answer)

finetuning_dataset_name = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_name)
print(finetuning_dataset)
```

# Instruction fine-tuning

Instruction fine-tuning( AKA instruction tune or instruction 
following LLMs) is a type of fine-tuning. Type includes: reasoning, routing, copilot, which is writing code, chat, different agents.

It teaches the model to follow instructions and behave more like a chatbot. Like GPT-3 to chatgpt.

## LLM Data Generation

Some existing data is ready as-is, online:
- FAQs
- Customer support conversations
- Slack messages

It's really this dialogue dataset or just instruction response datasets

if you don't have data, no problem:
- You can also convert your data into something that's more of a question-answer format or instruction following format by using a prompt template. i.e. README might be able to come be converted into a question-answer pair.
- You can also use another LLM to do this for you -a technique called Alpaca from Stanford that uses chat GPT to do this.
- You can use a pipeline of different open source models to do this as well.

## Instruction fineturning generalization

- It teaches this new behavior to the model. i.e. the answer of capital of france from base model vs finetuned model
- Can access model's pre-existing knowledge(learned in pre-existing pre-training step), generalize instructions to other data, not in finetuning dataset. i.e. code. this is actually findings from the chat GPT paper where the 
model can now answer questions about code even though they didn't have question answer pairs about that for their instruction fine-tuning. because it's really expensive to get programmers label data sets.

## Overview of Finetuning

It's a very iterative process to improve the model: **data prep - training - evaluation**. After you evaluate the model, you need to prep the data again to improve it. 

For different types of fine-tuning, data prep is really where you have differences.  training and evaluation is very similar.


## Code Samples

### Setup
```py
import os
import lamini

lamini.api_url = os.getenv("POWERML__PRODUCTION__URL")
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")

import itertools
import jsonlines

from datasets import load_dataset
from pprint import pprint

from llama import BasicModelRunner
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```
### Load instruction tuned dataset

```py
instruction_tuned_dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)

m = 5
print("Instruction-tuned dataset:")
top_m = list(itertools.islice(instruction_tuned_dataset, m))
for j in top_m:
  print(j)
```
### Two prompt templates

```py
prompt_template_with_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

prompt_template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
```

### Hydrate prompts (add data to prompts)

```py
processed_data = []
for j in top_m:
  if not j["input"]:
    processed_prompt = prompt_template_without_input.format(instruction=j["instruction"])
  else:
    processed_prompt = prompt_template_with_input.format(instruction=j["instruction"], input=j["input"])

  processed_data.append({"input": processed_prompt, "output": j["output"]})

pprint(processed_data[0])
```
### Save data to jsonl

```py
with jsonlines.open(f'alpaca_processed.jsonl', 'w') as writer:
    writer.write_all(processed_data)
```

### Compare non-instruction-tuned vs. instruction-tuned models

```py
dataset_path_hf = "lamini/alpaca"
dataset_hf = load_dataset(dataset_path_hf)
print(dataset_hf)

non_instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-hf")
non_instruct_output = non_instruct_model("Tell me how to train my dog to sit")
print("Not instruction-tuned output (Llama 2 Base):", non_instruct_output)

instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
instruct_output = instruct_model("Tell me how to train my dog to sit")
print("Instruction-tuned output (Llama 2): ", instruct_output)

chatgpt = BasicModelRunner("chat-gpt")
instruct_output_chatgpt = chatgpt("Tell me how to train my dog to sit")
print("Instruction-tuned output (ChatGPT): ", instruct_output_chatgpt)

```

### Try smaller models

```py
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer

#---------------------
finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_path)
print(finetuning_dataset)


test_sample = finetuning_dataset["test"][0]
print(test_sample)

print(inference(test_sample["question"], model, tokenizer))

```

### Compare to finetuned small model

```py
instruction_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")

print(inference(test_sample["question"], instruction_model, tokenizer))

```
