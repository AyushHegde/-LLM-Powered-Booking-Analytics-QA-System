from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from rag_qa import generate_answer

app = FastAPI()

# Load the dataset
data_path = "cleaned_hotel_bookings.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path)

# Load FAISS index
index_path = "hotel_bookings.index"
if not os.path.exists(index_path):
    raise FileNotFoundError(f"FAISS index file not found: {index_path}")

index = faiss.read_index(index_path)

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.get("/")
def home():
    return {"message": "Welcome to the AI/ML API!"}

@app.post("/analytics")
async def get_analytics():
    """Returns multiple hotel booking analytics."""
    
    # 1️⃣ Cancellation Rate
    cancellation_rate = df["is_canceled"].mean() * 100

    # 2️⃣ Revenue Trends Over Time (Fix Applied Here)
    df["arrival_date"] = pd.to_datetime(df["reservation_status_date"])
    df["total_revenue"] = df["stays_in_week_nights"] * df["adr"]
    monthly_revenue = df.groupby(df["arrival_date"].dt.to_period("M"))["total_revenue"].sum()
    monthly_revenue = {str(k): v for k, v in monthly_revenue.items()}  # ✅ Convert PeriodIndex to string

    # 3️⃣ Geographical Distribution
    country_counts = df["country"].value_counts().to_dict()

    # 4️⃣ Booking Lead Time Distribution
    lead_time_distribution = df["lead_time"].describe().to_dict()

    # Return all analytics as JSON
    return {
        "cancellation_rate": f"{cancellation_rate:.2f}%",
        "monthly_revenue": monthly_revenue,
        "top_countries": country_counts,
        "lead_time_distribution": lead_time_distribution
    }
from fastapi import FastAPI
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        answer = generate_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
