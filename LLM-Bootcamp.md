# Intro

LLM have been described as a blurry snapshot of the web.

- statistical approach to language understanding
  - not good but statistical or rules-based approach can still solve some problems
- Recurrent Neural Network(RNN)
  - consider prior steps in a sequence in the next prediction(progress word by word/during training)
- Transformer-Based
  - Attention is the ability to consider not only the source language but the context of the target language.
  - comparing to RNN, it processes a sentence all at once.

Q&A
parameter? - consider the parameters of a function, i.e. ax+b
word count?-> training data.

MLCC for more detail

# Generative AI

- Need to consider responsibilities
- Hallucination
  - training data
  - math
  - bias(popular answer)

# Prompt Enginnering

- provide more context
  - few shot(give more examples) vs one shot vs zero shot(reply on the model to complete the request)
- "role prompting", i.e. you are a MIT Mathematician
- ask for "chain-of-thought"
- "preamble" to improving fairness with context

# Approaches Beyond Prompts

A useful tool: makersuite

- Synthetic data  go/data-synth-app
- Fine turning: specific to a domain or purpose, e.g. legal, edu, retail. Existing LLM with additional training data specific to the domain. e.g. Med-Palm or go/duckie for internal Google knowledge
- adjusting part of the model by changing the weights of the model(by attaching context to the input prompt, e.g. you are a MIT Mathematician)
