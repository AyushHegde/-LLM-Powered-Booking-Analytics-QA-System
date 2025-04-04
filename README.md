# -LLM-Powered-Booking-Analytics-QA-System
Developed a system that processes hotel booking data, extracts insights, and enables  retrieval-augmented question answering (RAG). The system should provide analytics as  mentioned in below sections and answer user queries about the data.
# Hotel Booking Analytics & RAG-based QA System

This project combines traditional analytics with a Retrieval-Augmented Generation (RAG) based Question Answering (QA) system to help analyze hotel booking data and answer user questions using an LLM. Built using FastAPI, it supports both analytical insights and natural language queries.

---

## 🚀 Features

- ✅ Revenue trends, lead time stats, cancellation rates, country-wise distribution
- 🧐 Ask questions using natural language powered by LLM (RAG pipeline)
- 🔹 Semantic search using FAISS + Sentence Transformers
- 🚜 REST API for analytics and Q&A
- ⚡ CPU-compatible model (no GPU needed)

---

## 📁 Project Structure

```
solvei8-ai-ml-assignment/
├── main.py                # FastAPI server with API routes
├── rag_qa.py              # RAG-based QA logic
├── analytics.py           # Analytics calculations
├── cleaned_hotel_bookings.csv  # Cleaned dataset
├── hotel_bookings.index   # FAISS index
├── requirements.txt       # Python dependencies
└── README.md              # Project instructions and documentation
```

---

## 🚧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/solvei8-ai-ml-assignment.git
cd solvei8-ai-ml-assignment
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# Activate it
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API
```bash
python main.py
```

### 5. Test the API
Visit: [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

---

## 🔍 API Endpoints

### `POST /analytics`
Returns analytical insights like:
- Cancellation rate
- Monthly revenue trends
- Country distribution
- Lead time stats

### `POST /ask`
Ask a question based on the dataset.

**Example:**
```json
{
  "question": "What is the monthly revenue for July 2017?"
}
```

**Response:**
```json
{
  "answer": "The revenue for July 2017 was approximately 45,000 USD."
}
```

---

## 🌐 Sample Test Queries & Answers

| Query                                         | Expected Answer                          |
|----------------------------------------------|------------------------------------------|
| What is the cancellation rate?               | Around 37%                               |
| Which country has the most bookings?         | Portugal (PRT)                           |
| What is the average lead time?               | Approximately 85 days                   |
| Revenue trend for July 2017?                 | Highest of the year with ~$45,000        |

---

## 🎯 Performance Evaluation

### ✉ Accuracy
- Tested using manually validated queries and expected answers.
- Embedding similarity checks between query and top-k documents.

### ⏱ Response Time
- QA response time ranges between 4-10 seconds on CPU.
- Use time logging around `generate_answer()` to monitor latency.

---

## 🚀 Implementation Choices

- **Model:** Mistral 7B Instruct (used with Transformers pipeline)
- **Embedding:** all-MiniLM-L6-v2 (fast & light on CPU)
- **Vector DB:** FAISS for fast semantic search
- **API:** FastAPI with Swagger UI for easy testing

---

## 🚫 Challenges Faced

- Long loading time of LLM on CPU (mitigated with smaller `max_new_tokens`)
- Some questions too vague or outside dataset scope
- HuggingFace rate limits (solved with token auth)

---

## 💾 To Do

- [ ] Add Dockerfile for easy deployment
- [ ] Add query history logging and metrics
- [ ] Use quantized model to speed up response time

---

## 🚪 License

This project is open-source and free to use under the MIT License.

