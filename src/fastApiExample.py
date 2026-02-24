
from fastapi import FastAPI
import uvicorn
import nest_asyncio
import threading

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello! FastAPI is running from a notebook."}

@app.post("/predict")
def predict(payload: dict):
    text = (payload.get("text") or "").lower()
    is_perishable = any(k in text for k in ["milk", "yogurt", "meat", "fish", "cheese"])
    return {"input": payload, "is_perishable": is_perishable}

def run():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

nest_asyncio.apply()

thread = threading.Thread(target=run, daemon=True)
thread.start()

print("FastAPI running at http://127.0.0.1:8000")
print("Docs at http://127.0.0.1:8000/docs")





