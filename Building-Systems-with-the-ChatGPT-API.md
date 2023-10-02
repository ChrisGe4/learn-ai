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
   |
   v
`assistant` LLM response
   | ^
   v | 
`user` prompts    
