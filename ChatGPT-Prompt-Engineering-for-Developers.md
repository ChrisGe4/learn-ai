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
> Identify types of emotions
```
Identify a list of emotions that the writer of the \
following review is expressing. Include no more than \
five items in the list. Format your answer as a list of \
lower-case words separated by commas.

Review text: '''{lamp_review}'''
```
    
> Identify anger

```
Is the writer of the following review expressing anger?\
The review is delimited with triple backticks. \
Give your answer as either yes or no.

Review text: '''{lamp_review}'''
"""
```
- **Extract Information**:
  > product and company name from customer reviews
```
  Identify the following items from the review text: 
  - Item purchased by reviewer
  - Company that made the item
  
  The review is delimited with triple backticks. \
  Format your response as a JSON object with \
  "Item" and "Brand" as the keys. 
  If the information isn't present, use "unknown" \
  as the value.
  Make your response as short as possible.
    
  Review text: '''{lamp_review}'''
```
  > Doing multiple tasks at once
  ```
  Identify the following items from the review text: 
  - Sentiment (positive or negative)
  - Is the reviewer expressing anger? (true or false)
  - Item purchased by reviewer
  - Company that made the item
  
  The review is delimited with triple backticks. \
  Format your response as a JSON object with \
  "Sentiment", "Anger", "Item" and "Brand" as the keys.
  If the information isn't present, use "unknown" \
  as the value.
  Make your response as short as possible.
  Format the Anger value as a boolean.
  
  Review text: '''{lamp_review}'''
  ```
- Inferring topics
```
Determine five topics that are being discussed in the \
following text, which is delimited by triple backticks.

Make each item one or two words long. 

Format your response as a list of items separated by commas.

Text sample: '''{story}'''
```
- Make a news alert for certain topics
```
Determine whether each item in the following list of \
topics is a topic in the text below, which
is delimited with triple backticks.

Give your answer as list with 0 or 1 for each topic.\

List of topics: {", ".join(topic_list)}

Text sample: '''{story}'''
```
# Transforming

## Translation
```
Translate the following English text to Spanish:
'''TEXT'''
```

```
Tell me which language this is: 
'''TEXT'''
```

```
Translate the following  text to French and Spanish
and English pirate: \
'''TEXT'''
```

```
Translate the following text to Spanish in both the \
formal and informal forms:  \
'''TEXT'''
```
### Universal Translator

Use case: Imagine you are in charge of IT at a large multinational e-commerce company. Users are messaging you with IT issues in all their native languages. Your staff is from all over the world and speaks only their native languages. You need a universal translator!

```
user_messages = [
  "La performance du système est plus lente que d'habitude.",  # System performance is slower than normal         
  "Mi monitor tiene píxeles que no se iluminan.",              # My monitor has pixels that are not lighting
  "Il mio mouse non funziona",                                 # My mouse is not working
  "Mój klawisz Ctrl jest zepsuty",                             # My keyboard has a broken control key
  "我的屏幕在闪烁"                                               # My screen is flashing
]
```
```
for issue in user_messages:
    prompt = f"Tell me what language this is: ```{issue}```"
    lang = get_completion(prompt)
    print(f"Original message ({lang}): {issue}")

    prompt = f"""
    Translate the following  text to English \
    and Korean: ```{issue}```
    """
    response = get_completion(prompt)
    print(response, "\n")
```

## Tone Transformation
```
Translate the following from slang to a business letter: 
'Dude, This is Joe, check out this spec on this standing lamp.'
"""
```
Note: It will be expanded with much more content as a real email.

## Format Conversion
i.e. Dict -> Json ->html
```
data_json = { "resturant employees" :[ 
    {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob", "email":"bob32@gmail.com"},
    {"name":"Jai", "email":"jai87@gmail.com"}
]}

prompt = f"""
Translate the following python dictionary from JSON to an HTML \
table with column headers and title: {data_json}
"""
```
## Spellcheck/Grammar check
proofread and correct
```
Proofread and correct the following text \
and rewrite the corrected version. If you don't find \
and errors, just say "No errors found". Don't use \
any punctuation around the text:
    '''{t}'''
```

```
proofread and correct this review. Make it more compelling. 
Ensure it follows APA style guide and targets an advanced reader. 
Output in markdown format.
Text: '''{text}'''
```
# Expanding
## Customize the automated reply to a customer email
```
You are a customer service AI assistant.
Your task is to send an email reply to a valued customer.
Given the customer email delimited by ''', \
Generate a reply to thank the customer for their review.
If the sentiment is positive or neutral, thank them for \
their review.
If the sentiment is negative, apologize and suggest that \
they can reach out to customer service. 
Make sure to use specific details from the review.
Write in a concise and professional tone.
Sign the email as `AI customer agent`.
Customer review: '''{review}'''
Review sentiment: {sentiment} # not necessary, we can let ai to extract it
```
***Temperature***
0-1
0: more reliable and predictable
1: more creative and wider variaty of output(random)

# Chatbot

ChatGPT can take a series of messages as input and then return a model generated message as output.
The following examples use a structured data to specify the role of each message.
*Note*: LangChain provides fine grained control of the input.

```
messages =  [  
{'role':'system', 'content':'You are an assistant that speaks like Shakespeare.'},    
{'role':'user', 'content':'tell me a joke'},   
{'role':'assistant', 'content':'Why did the chicken cross the road'},   
{'role':'user', 'content':'I don\'t know'}  ]
```
prompt runs in transaction. You need to provide the earlier exchanges in the input to the model to interact continuesly.

# Order Bot

Reuse a predefined context in each interaction.
```
context = [ {'role':'system', 'content':"""
You are OrderBot, an automated service to collect orders for a pizza restaurant. \
You first greet the customer, then collects the order, \
and then asks if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final \
time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Finally you collect the payment.\
Make sure to clarify all options, extras and sizes to uniquely \
identify the item from the menu.\
You respond in a short, very conversational friendly style. \
The menu includes \
pepperoni pizza  12.95, 10.00, 7.00 \
cheese pizza   10.95, 9.25, 6.50 \
eggplant pizza   11.95, 9.75, 6.75 \
fries 4.50, 3.50 \
greek salad 7.25 \
Toppings: \
extra cheese 2.00, \
mushrooms 1.50 \
sausage 3.00 \
canadian bacon 3.50 \
AI sauce 1.50 \
peppers 1.00 \
Drinks: \
coke 3.00, 2.00, 1.00 \
sprite 3.00, 2.00, 1.00 \
bottled water 5.00 \
"""} ]  # accumulate messages
```
