#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from fastapi import FastAPI
import uvicorn
import nest_asyncio
import threading


model_name="vennify/t5-base-grammar-correction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello! FastAPI is running from a notebook."}
@app.post("/predict")
def predict(payload):
    return grammarCorrection(payload)

def grammarCorrection(text):
     prompt = "grammar: " + text
     inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)

     outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
     corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
     return corrected
def run():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

nest_asyncio.apply()

thread = threading.Thread(target=run, daemon=True)
thread.start()

print("FastAPI running at http://127.0.0.1:8000")
print("Docs at http://127.0.0.1:8000/docs")


# In[ ]:




