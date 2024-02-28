# Keyword Search

## Setup

Optional: To run locally, first do 

```py
# !pip install cohere
# !pip install weaviate-client
```
ENV

```py
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) #read local .env file
```

imporing Weaviate to access the Wikipedia database.  Weaviate is an open source database. It has keyword search capabilities and also vector search capabilities that rely on language models.

```py
import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])

client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'], # "https://cohere-demo.weaviate.network/" contains 10 million recoreds in 10 different languages, so 1 mil in eng
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)

client.is_ready() 
```

## Keyword Search

![dia](doc-data/keyword-search.png)

first stage, the retrieval, commonly uses the BM25 algorithm to score the documents in the archive versus the query. The implementation of the first stage retrieval often 
contains an inverted index.

Limitations: if we search a doc archive that has this other document that answers it exactly, but it uses different keywors, (e.g. strong pain in the side of the head vs sharp temple headache), keyword search is not going to
be able to retrieve this doc. LLM can help, bc they can look at the general meaning and they're able to retrieve a doc like this

bm_25 is the keyword search or lexical search algorithm commonly used, and it scores the documents in the archive versus the query based on a specific formula that
looks at the count of the shard words between the query and each document.

```py
def keyword_search(query,results_lang='en',properties = ["title","url","text"],num_results=3):
  where_filter = {
  "path":["lang"],
  "operator":"Equal",
  "valueString": results_lang
  }

response = (
        client.query.get("Articles", properties)
        .with_bm25(
            query=query
        )
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
        )

result = response['data']['Get']['Articles']
return result

query = "What is the most viewed televised event?"
keyword_search_results = keyword_search(query)
print(keyword_search_results)

```
### Try modifying the search options
- Other languages to try: `en, de, fr, es, it, ja, ar, zh, ko, hi`

```py
properties = ["text", "title", "url", 
             "views", "lang"]

def print_result(result):
    """ Print results with colorful formatting """
    for i,item in enumerate(result):
        print(f'item {i}')
        for key in item.keys():
            print(f"{key}:{item.get(key)}")
            print()
        print()

print_result(keyword_search_results)
```

```py
query = "What is the most viewed televised event?"
keyword_search_results = keyword_search(query, results_lang='de')
print_result(keyword_search_results)
```

