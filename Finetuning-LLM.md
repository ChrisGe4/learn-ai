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
