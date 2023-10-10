# Language Models, the Chat Format and Tokens

Mentioned Concepts:
- Supervised learning (x->y) to repeateddly predict the next word.

Two types of LLM
- Base LLM: predicts next word, based on text training data.
- Instruction tuned LLM: Tries to follow instructions (ChatGPT)

From a Base LLM to an Instruction Tuned LLM
1. first train a Base LLM on a lot of data
1. then further train the model by fine-tuning it on a smaller set of examples, where the 
output follows an input instruction. And so, for example, you may 
have contractors help you write a lot of examples of an instruction, 
and then a good response to an instruction. And that creates a training set to carry 
out this additional fine-tuning. So that learns to predict what is 
the next word if it's trying to follow an instruction.
1. After that, to improve the quality of the LLM's output, a common process now is to obtain human ratings of the quality of many different 
LLM outputs on criteria, such as whether the output is helpful, honest, and harmless.
1. further tune the LLM to increase the probability of its generating the more highly rated outputs. The most common technique 
to do this is RLHF, which stands for Reinforcement Learning from Human Feedback.

- Base LLM -> months
- Base LLM to Instruction Tuned LLM can be done in maybe days.

LLM doesn't actually repeatedly predict the next word, it instead repeatedly predicts the next token. 

What an LLM actually does is it will take a sequence of characters, like "Learning new things is fun!", and group the characters together to form tokens that 
comprise **commonly occurring sequences of characters**.  Token can be one word, or one word in a space, or an exclamation mark. Less frequently used words will be broken down to tokens, such as "prompting"-> "'prom", "pt", and "ing" because those three are commonly occurring sequences of letters. **A trick is to add dashes between letters if you want to tokenize letters** (i.e to reverse letters in a word)

For English, 1 token is around 4 chars, or 3/4 of a word.

Token limits = input `context` + output `completion`

# System, User, and Assistant Messages

```
messages =  [  
{'role':'system',
 'content':'All your responses must be \
one sentence long.'},    
{'role':'user',
 'content':'write me a story about a happy carrot'},  
]
```

`system` - sets the tone or overall behavior of assistant
 
   ↓

`assistant` LLM response - (you can also put the things previous said here)

   ⇅
   
`user` prompts    

Above is how chat format works. System message set the overall behavior. Can add more contrains to it, like "one sentence long"

**Note**: token number embedded into the response.

- supervised learning: get labeled data(1 month) -> train model on data(3 monthes)-> deploy&call model(3 monthes)
- prompt-based AI: specify prompt(minutes/hours)->call model(m/h)

**Caveat**: currently not working well on many unstructured data applications, includeing specifially text applications and vision applications. This recipe doesn't work for structured data applications, meaning ML applications on tabular data with lots of numerical values in Excel spreadsheets.

# Classification

example:

```py
// a way to separate different parts of an instruction or output, and it helps the model kind of determine the different sections.
delimiter = "####"

system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with \
{delimiter} characters.
Classify each query into a primary category \
and a secondary category. 
Provide your output in json format with the \
keys: primary and secondary.

Primary categories: Billing, Technical Support, \
Account Management, or General Inquiry.

Billing secondary categories:
Unsubscribe or upgrade
Add a payment method
Explanation for charge
Dispute a charge

Technical Support secondary categories:
General troubleshooting
Device compatibility
Software updates

Account Management secondary categories:
Password reset
Update personal information
Close account
Account security

General Inquiry secondary categories:
Product information
Pricing
Feedback
Speak to a human

"""
```
The prompt for classification is:

**Classify each query into a primary category \
and a secondary category. 
Provide your output in json format with the \
keys: primary and secondary.**

```python
user_message = f"""\
I want you to delete my profile and all of my user data"""
messages =  [  
{'role':'system', 
 'content': system_message},    
{'role':'user', 
 'content': f"{delimiter}{user_message}{delimiter}"},  
]
```
