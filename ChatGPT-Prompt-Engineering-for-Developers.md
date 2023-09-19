## Tactics 1: Use delimiters to clearly indicate distinct parts of the input
* Delimiters can be anything like: ```, """, < >, <tag> </tag>
```
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
```
## Tactic 2: Ask for a structured output
- JSON, HTML
```
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
```
## Tactic 3: Ask the model to check whether conditions are satisfied
- Check assumptions required to do the task

```
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

Tactic 4: "Few-shot" prompting
Give successful examples of completing tasks
Your task is to answer in a consistent style.

You: xxx?
Me: xxx
You: xxx?
```

## Tactic 5: Specify the steps required to complete a task
```
Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.

Text:
```{text}```
```

```
Ask for output in a specified format
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the 
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in Italian summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
```

## Tactic 6: Instruct the model to work out its own solution before rushing to a conclusion
```
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.

Use the following format:
Question:
```
question here
```
Student's solution:
```
student's solution here
```
Actual solution:
```
steps to work out the solution and your solution here
```
Is the student's solution the same as actual solution \
just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect
```

Question:
```
CONTENT
``` 
Student's solution:
```
CONTENT
```
Actual solution:
```

## Tactic 7: Reducing hallucinations by 
- Frist find relevant information
- Then answer the question based on the relevant information


**A note about the backslash**
In the course, we are using a backslash \ to make the text fit on the screen without inserting newline '\n' characters.
GPT-3 isn't really affected whether you insert newline characters or not. But when working with LLMs in general, you may consider whether newline characters in your prompt may affect the model's performance.

# Iterative

```
Your task is to help a marketing team create a descritpion for a retail website of a product based on a technical fact sheet.

Write a product description based on the information provided in the technical specification delimited by triple quotes.

<Do iteration here>
 

Technical specifications: """{fact_sheet_chair}"""
```
- Issue 1: the text is too long -> Limit the number of words/sentences/characters.
```
Use at most 50 words/use 3 sentances/ 200 characters
```
- Issue 2: Text focuses on the wrong details -> Ask it to focus on the aspects that are relevant to the intended audience.
```
The description is intended for furniture retailers, 
so should be technical in nature and focus on the 
materials the product is constructed from.

optional:
At the end of the description, include every 7-character 
Product ID in the technical specification.
```
- Issue 3. Description needs a table of dimensions -> Ask it to extract information and organize it in a table.

```
After the description, include a table that gives the 
product's dimensions. The table should have two columns.
In the first column include the name of the dimension. 
In the second column include the measurements in inches only.

Give the table the title 'Product Dimensions'.

Format everything as HTML that can be used in a website. 
Place the description in a <div> element.
```
# Summarizing

- Summarize with a word/sentence/character limit
```
Your task is to generate a short summary of a product \
review from an ecommerce site. 

Summarize the review below, delimited by triple 
quotes, in at most 30 words. 

Review: """{prod_review}"""
```
-  Summarize with a focus
```
Your task is to generate a short summary of a product \
review from an ecommerce site. 

Summarize the review below, delimited by triple 
quotes, in at most 30 words, and focusing on any aspects \
* that mention shipping and delivery of the product.
* that are relevant to the price and perceived value. 
Review: """{prod_review}"""
```
- Summaries include topics that are not related to the topic of focus. Try "extract" instead of "summarize"
Try "extract" instead of "summarize"
```
Your task is to extract relevant information from \ 
a product review from an ecommerce site to give \
feedback to the Shipping department. 

From the review below, delimited by triple quotes \
extract the information relevant to shipping and \ 
delivery. Limit to 30 words. 

Review: """{prod_review}"""
```
- Able to summarize multiple product reviews

# Inferring
- Sentiment (positive/negative)
```
What is the sentiment of the following product review, 
which is delimited with triple single quotes?

[optional]
Give your answer as a single word, either "positive" \
or "negative".

Review text: '''{lamp_review}'''
```
