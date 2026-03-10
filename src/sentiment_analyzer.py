#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import transformers
from transformers import pipeline

#pip install transformers torch 
#version 5.2.0
#print(transformers.__version__)
# Load the model
# This should be done only once during application startup
# Initializing the model for every request is not recommended due to performance overhead
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)



# In[ ]:


import nest_asyncio
from typing import List
nest_asyncio.apply()
import io
from fastapi import FastAPI,HTTPException
from fastapi.responses import RedirectResponse
import uvicorn


app = FastAPI(docs_url="/docs")

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="http://127.0.0.1:8080/docs")

@app.post("/find_sentiments" ,summary="Find sentiments for list of texts",
    description="This API accepts an array of strings and returns sentiment results.")
def find_sentiments(items: List[str]):
    print("Inside the smentiments process_items")
    return findsentiments(items)

config = uvicorn.Config(app, host="127.0.0.1",port=8080)
server = uvicorn.Server(config)
await server.serve()


# In[ ]:


# Determine sentiment category based on model confidence scores
def convert_sentiment(label, confidence):
    LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
    }

    sentiment = LABEL_MAP.get(label, "Unknown")
    weight = round(confidence * 100, 2)

    if confidence < 0.60:
        return {
            "sentiment": sentiment,
            "weightage": weight,
            "strength": "Low (Mixed)"
        }
    elif confidence < 0.75:
        return {
            "sentiment": sentiment,
            "weightage": weight,
            "strength": "Medium"
        }
    else:
        return {
            "sentiment": sentiment,
            "weightage": weight,
            "strength": "High"
        }


# In[ ]:


def findsentiments(texts):
    results_list = []
    results = sentiment_pipeline(texts)
    #print("The result is " , results)
    # Display results
    for text, result in zip(texts, results):
         results_list.append(
         convert_sentiment(result["label"], result["score"])
        )
    return(results_list)


# In[ ]:




