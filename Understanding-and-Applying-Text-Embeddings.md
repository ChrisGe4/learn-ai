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
**Note**: use the original for real use case.

