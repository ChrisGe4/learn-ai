# Getting Start

## Setup

### Install Vertex AI

```bash
!pip install google-cloud-aiplatform
```
Then

```python
from utils import authenticate
credentials, PROJECT_ID = authenticate() # Get credentials and project ID
REGION = 'us-central1'
---
import vertexai
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)

```
### Use the embeddings model

```python
from vertexai.language_models import TextEmbeddingModel

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")

```

Note: **textembedding-gecko** is the name for the model that supports text embeddings. Text embeddings are a NLP technique that converts textual data into numerical vectors that can be processed by machine learning algorithms, especially large models. These vector representations are designed to capture the semantic meaning and context of the words they represent.
There are a few versions available for embeddings. textembedding-gecko@latest is the newest version with enhanced AI quality versus the current stable version textembedding-gecko@001, and textembedding-gecko-multilingual@latest is a model optimized for a wide range of non-English languages.

#### Generate a word/sentence embedding

```python
embedding = embedding_model.get_embeddings(["life"])

// The returned object is a list with a single TextEmbedding object.
// The TextEmbedding.values field stores the embeddings in a Python list.

vector = embedding[0].values
print(f"Length = {len(vector)}")  # 768 dimensions
print(vector[:10])  

```

#### Similarity

- Calculate the similarity between two sentences as a number between 0 and 1.

```python
from sklearn.metrics.pairwise import cosine_similarity

emb_1 = embedding_model.get_embeddings(
    ["What is the meaning of life?"]) # 42!

emb_2 = embedding_model.get_embeddings(
    ["How does one spend their time well on Earth?"])

emb_3 = embedding_model.get_embeddings(
    ["Would you like a salad?"])

vec_1 = [emb_1[0].values]
vec_2 = [emb_2[0].values]
vec_3 = [emb_3[0].values]

# Note: the reason we wrap the embeddings (a Python list) in another list is because the cosine_similarity function expects either a 2D numpy array or a list of lists.

print(cosine_similarity(vec_1,vec_2))

# Note: the value is not vary significantly, b/c of cosine.

```

#### From word to sentence embeddings
- One possible way to calculate sentence embeddings from word embeddings is to take the average of the word embeddings.
  This ignores word order and context, so two sentences with different meanings, but the same set of words will end up with the same sentence embedding.
- As to sentence, these sentence embeddings account for word order and context.


# Understanding Text Embeddings

A way of represening data as points in space where the locations are semantically meaningful.

- Simple method: Embed each word separately, and take a sum of mean of all the word embeddings.
- Modern embeddings: use a transformer neural network to compute a context-aware representation of each word, then take an average of the context-aware representations.
- Moderner: Compute embeddings for each token (e.g., sub-words) rather thatn word. Enables algorithm to work even for novel words and misspelt words. You can throw any strings, and still get the embeddings.

## Training the transformer network - **contrastive learning**

Given a dataset of pairs of "similar" sentences, tune neural network to move similar sentences' embeddings together, and dissimilar sentences' embeddings apart. 
- Find similar sentences: data set with similar sentences; Q - A pairs, to tell the algo to push the Q to the A closer to the embedding of the Q of the A.
- dissimilar: random sentences.

Note: Researchers still playing around this receipt with varients, so it gets improved every a few months.

## Multi-modal embeddings

Text - Pictures

# Visualizing Embaddings

Use principal component analysis (PCA - is a dimensionality reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.). You can learn more about PCA in [this video](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/73zWO/reducing-the-number-of-features-optional) from the Machine Learning Specialization.

Lost data along the way but makes it easy to plot.

```python

from sklearn.decomposition import PCA

# Perform PCA for 2D visualization
PCA_model = PCA(n_components = 2)
PCA_model.fit(embeddings_array)
new_values = PCA_model.transform(embeddings_array)

print("Shape: " + str(new_values.shape))
print(new_values)

```

```python
import matplotlib.pyplot as plt
import mplcursors
%matplotlib ipympl

from utils import plot_2D
plot_2D(new_values[:,0], new_values[:,1], input_text_lst_news)
```
**Note**: use the original dimension for real use case.

# Applications of Embeddings

- Cluster the embeddings
- Anomaly/Outlier detection
- Classification

Sample code - https://learn.deeplearning.ai/google-cloud-vertex-ai/lesson/5/applications-of-embeddings


#### Project environment setup

- Load credentials and relevant Python Libraries
```
from utils import authenticate
credentials, PROJECT_ID = authenticate()

REGION = 'us-central1'

import vertexai
vertexai.init(project=PROJECT_ID, 
              location=REGION, 
              credentials = credentials)
```
#### Load Stack Overflow questions and answers from BigQuery
- BigQuery is Google Cloud's serverless data warehouse.
- We'll get the first 500 posts (questions and answers) for each programming language: Python, HTML, R, and CSS.
```
from google.cloud import bigquery
import pandas as pd

def run_bq_query(sql):

    # Create BQ client
    bq_client = bigquery.Client(project = PROJECT_ID, 
                                credentials = credentials)

    # Try dry run before executing query to catch any errors
    job_config = bigquery.QueryJobConfig(dry_run=True, 
                                         use_query_cache=False)
    bq_client.query(sql, job_config=job_config)

    # If dry run succeeds without errors, proceed to run query
    job_config = bigquery.QueryJobConfig()
    client_result = bq_client.query(sql, 
                                    job_config=job_config)

    job_id = client_result.job_id

    # Wait for query/job to finish running. then get & return data frame
    df = client_result.result().to_arrow().to_pandas()
    print(f"Finished job_id: {job_id}")
    return df
 
# define list of programming language tags we want to query

language_list = ["python", "html", "r", "css"]

so_df = pd.DataFrame()

for language in language_list:
    
    print(f"generating {language} dataframe")
    
    query = f"""
    SELECT
        CONCAT(q.title, q.body) as input_text,
        a.body AS output_text
    FROM
        `bigquery-public-data.stackoverflow.posts_questions` q
    JOIN
        `bigquery-public-data.stackoverflow.posts_answers` a
    ON
        q.accepted_answer_id = a.id
    WHERE 
        q.accepted_answer_id IS NOT NULL AND 
        REGEXP_CONTAINS(q.tags, "{language}") AND
        a.creation_date >= "2020-01-01"
    LIMIT 
        500
    """

    
    language_df = run_bq_query(query)
    language_df["category"] = language
    so_df = pd.concat([so_df, language_df], 
                      ignore_index = True) 
```
- You can reuse the above code to run your own queries if you are using Google Cloud's BigQuery service.
- In this classroom, if you run into any issues, you can load the same data from a csv file.
```
# Run this cell if you get any errors or you don't want to wait for the query to be completed
# so_df = pd.read_csv('so_database_app.csv')

so_df
```
#### Generate text embeddings
- To generate embeddings for a dataset of texts, we'll need to group the sentences together in batches and send batches of texts to the model.
- The API currently can take batches of up to 5 pieces of text per API call.
```
from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")

import time
import numpy as np

# Generator function to yield batches of sentences

def generate_batches(sentences, batch_size = 5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]

so_questions = so_df[0:200].input_text.tolist() 
batches = generate_batches(sentences = so_questions)

batch = next(batches)
len(batch)
```
#### Get embeddings on a batch of data
- This helper function calls `model.get_embeddings()` on the batch of data, and returns a list containing the embeddings for each text in that batch.
```
def encode_texts_to_embeddings(sentences):
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]

batch_embeddings = encode_texts_to_embeddings(batch)

f"{len(batch_embeddings)} embeddings of size \
{len(batch_embeddings[0])}"
```
#### Code for getting data on an entire data set
- Most API services have rate limits, so we've provided a helper function (in utils.py) that you could use to wait in-between API calls.
- If the code was not designed to wait in-between API calls, you may not receive embeddings for all batches of text.
- This particular service can handle 20 calls per minute.  In calls per second, that's 20 calls divided by 60 seconds, or `20/60`.

```Python
from utils import encode_text_to_embedding_batched

so_questions = so_df.input_text.tolist()
question_embeddings = encode_text_to_embedding_batched(
                            sentences=so_questions,
                            api_calls_per_second = 20/60, 
                            batch_size = 5)
```

In order to handle limits of this classroom environment, we're not going to run this code to embed all of the data. But you can adapt this code for your own projects and datasets.

#### Load the data from file
- We'll load the stack overflow questions, answers, and category labels (Python, HTML, R, CSS) from a .csv file.
- We'll load the embeddings of the questions (which we've precomputed with batched calls to `model.get_embeddings()`), from a pickle file.
```
so_df = pd.read_csv('so_database_app.csv')
so_df.head()

import pickle

with open('question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)

print("Shape: " + str(question_embeddings.shape))
print(question_embeddings)
```
#### Cluster the embeddings of the Stack Overflow questions
```
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

clustering_dataset = question_embeddings[:1000]

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, 
                random_state=0, 
                n_init = 'auto').fit(clustering_dataset)

kmeans_labels = kmeans.labels_

PCA_model = PCA(n_components=2)
PCA_model.fit(clustering_dataset)
new_values = PCA_model.transform(clustering_dataset)

import matplotlib.pyplot as plt
import mplcursors
%matplotlib ipympl

from utils import clusters_2D
clusters_2D(x_values = new_values[:,0], y_values = new_values[:,1], 
            labels = so_df[:1000], kmeans_labels = kmeans_labels)
```
- Clustering is able to identify two distinct clusters of HTML or Python related questions, without being given the category labels (HTML or Python).

## Anomaly / Outlier detection

- We can add an anomalous piece of text and check if the outlier (anomaly) detection algorithm (Isolation Forest) can identify it as an outlier (anomaly), based on its embedding.
```
from sklearn.ensemble import IsolationForest

input_text = """I am making cookies but don't 
                remember the correct ingredient proportions. 
                I have been unable to find 
                anything on the web."""

emb = model.get_embeddings([input_text])[0].values

embeddings_l = question_embeddings.tolist()
embeddings_l.append(emb)

embeddings_array = np.array(embeddings_l)

print("Shape: " + str(embeddings_array.shape))
print(embeddings_array)

# Add the outlier text to the end of the stack overflow dataframe
so_df = pd.read_csv('so_database_app.csv')
new_row = pd.Series([input_text, None, "baking"], 
                    index=so_df.columns)
so_df.loc[len(so_df)+1] = new_row
so_df.tail()
```
#### Use Isolation Forest to identify potential outliers

- `IsolationForest` classifier will predict `-1` for potential outliers, and `1` for non-outliers.
- You can inspect the rows that were predicted to be potential outliers and verify that the question about baking is predicted to be an outlier.
```
clf = IsolationForest(contamination=0.005, 
                      random_state = 2) 

preds = clf.fit_predict(embeddings_array)

print(f"{len(preds)} predictions. Set of possible values: {set(preds)}")

so_df.loc[preds == -1]
```
#### Remove the outlier about baking
```
so_df = so_df.drop(so_df.index[-1])

so_df
```
## Classification
- Train a random forest model to classify the category of a Stack Overflow question (as either Python, R, HTML or CSS).
```
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# re-load the dataset from file
so_df = pd.read_csv('so_database_app.csv')
X = question_embeddings
X.shape

y = so_df['category'].values
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 2)

clf = RandomForestClassifier(n_estimators=200)

clf.fit(X_train, y_train)
```
#### You can check the predictions on a few questions from the test set
```
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) # compute accuracy
print("Accuracy:", accuracy)
```
#### Try out the classifier on some questions
```
# choose a number between 0 and 1999
i = 2
label = so_df.loc[i,'category']
question = so_df.loc[i,'input_text']

# get the embedding of this question and predict its category
question_embedding = model.get_embeddings([question])[0].values
pred = clf.predict([question_embedding])

print(f"For question {i}, the prediction is `{pred[0]}`")
print(f"The actual label is `{label}`")
print("The question text is:")
print("-"*50)
print(question)
```


# Text Generation with Vertex AI

Models:
- text-bison@001: trained to handle a variety of natural language tasks
- chat-bison@001: for multi-turn dialogue

## For more predictability of the language model's response, you can also ask the language model to choose among a list of answers and then elaborate on its answer.

from

```
I'm a high school student. \
Recommend me a programming activity to improve my skills.
```

to

```
I'm a high school student. \
Which of these activities do you suggest and why:
a) learn Python
b) learn Javascript
c) learn Fortran
```
## Extract information and format it as a table

```
<a long text>

Extract the characters, their jobs \
and the actors who played them from the above message as a table
```

## Adjusting Creativity/Randomness

The decoding strategy applies top_k, then top_p, then temperature (in that order).

If you want to adjust top_p and top_k and see different results, remember to set **temperature to be greater than zero**, otherwise the model will always choose the token with the highest probability.


### Top K

Sample from tokens withe the top k probabilities. Works well with several words are fairly likely, not very well when the probability distribution is skewed, which means one word is very likely and other words are not very likely.

- The default value for top_k is 40.
- You can set top_k to values between 1 and 40.

### Top P

Sample the minimum set of tokens whose probabilities add up to probability p or greater.

- The default value for top_p is 0.95.


### Temperature
Use the probabilities to sample a random token.  Greedy decoding vs Random sample.  (Take autocomplete as an example. )

- You can control the behavior of the language model's decoding strategy by adjusting the temperature, top-k, and top-n parameters.
- For tasks for which you want the model to consistently/reliable output the same result for the same input, (such as classification or information extraction), set temperature to zero.
- For tasks where you desire more creativity, such as brainstorming, summarization, choose a higher temperature (up to 1).

Concepts:
- logits: (googled) the raw outputs of a model before they are transformed into probabilities. Specifically, logits are the unnormalized outputs of the last layer of a neural network.
- softmax: (googled) is an activation function that outputs the probability for each class and these probabilities will sum up to one.
- softmax with temperature: (googled) The temperature parameter is defined as the inverse of the scaling factor used to adjust the logits before the softmax function is applied. When the temperature is set to a low value, the probabilities of the predicted words are sharpened, which means that the most likely word is selected with a higher probability.

**Note**: start with 0.2 temperature.

# Semantic Search, Building a Q&A System

## Grounding LLMs

out-of-box LLMs aren't connected to the real world. Response would depend on the knowledge cut-off. You can't stuff everything in the prompt b/c it will soon exceed the token limit.

Grouding LLM: 
- Access information outside of training data
- Integrate with existing IT systems, databases and business data, to trace the lineage responses/the origin of the answers. 
- Mitigate risk of hallucinations(model provides responses that seem plausible, but not grouded in reality and fractual accurate).
