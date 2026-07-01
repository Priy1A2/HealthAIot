# HealthAIoT: Client-Side TinyML Inference & Agentic RAG Smart Healthcare Platform

An end-to-end implementation of the core concepts from the **HealthAIoT** research paper, adapted for a highly scalable, free-tier cloud architecture. This platform evaluates diabetes mellitus risk using a hybrid cloud-edge system: running real-time, zero-latency inference directly in the browser via WebAssembly (TinyML) and serving clinically validated, semantic explanations using an Agentic Retrieval-Augmented Generation (RAG) backend.

---

## 🚀 Key Features

- **On-Device TinyML Inference:** Executes a trained, scaled LightGBM classifier directly inside the browser using **ONNX Runtime Web**, providing instant risk assessments while ensuring complete patient data privacy and zero hosting costs.
- **Semantic Vector Search:** Integrates a serverless **Supabase PostgreSQL** database with the `pgvector` extension, enabling semantic retrieval of clinical guidelines (ADA Standards of Care) via a custom cosine similarity matching algorithm.
- **Compliant Generative Agent:** Leverages the **Gemini 2.5 Flash** model (via the unified `google-genai` SDK) to synthesize personalized preventive action plans, strictly bounded by safe clinical system instructions to prevent medical hallucinations.
- **Production-Ready Containerization:** Standardized using **Docker** and configured to run concurrently under **Gunicorn** to handle high-concurrency production workloads.

---

## 📐 Architecture Overview

```text
  ┌────────────────────────────────────────────────────────────────────────┐
  │                              CLIENT LAYER                              │
  │  ┌───────────────────────┐            ┌─────────────────────────────┐  │
  │  │ Smartwatch / Wearable │ ─────────> │     User Web Browser        │  │
  │  └───────────────────────┘            │ - ONNX Runtime Web (TinyML) │  │
  │                                       │ - Immediate Risk Prediction │  │
  │                                       └──────────────┬──────────────┘  │
  └──────────────────────────────────────────────────────┼─────────────────┘
                                                         │ Anonymized Score
                                                         │ & Metrics
                                                         ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                            BACKEND SERVICES                            │
  │  ┌───────────────────────┐            ┌─────────────────────────────┐  │
  │  │     Flask Backend     │ <────────> │    Supabase Vector DB       │  │
  │  │ (Dockerized / Gunicorn)│            │  (PostgreSQL + pgvector)    │  │
  │  └───────────┬───────────┘            └─────────────────────────────┘  │
  │              │                                                         │
  │              ▼ Secure Prompt Context                                   │
  │  ┌───────────────────────┐                                             │
  │  │    Gemini 2.5 Flash   │                                             │
  │  │ (Clinical AI Explainer)│                                            │
  │  └───────────────────────┘                                             │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

| File | Description |
|---|---|
| `index.html` | Responsive frontend dashboard built with Tailwind CSS and ONNX Runtime Web. |
| `server.py` | Flask backend API managing the RAG orchestration, Supabase querying, and Gemini API calls. |
| `seed_db.py` | Database populator script that uses local SentenceTransformers (`all-MiniLM-L6-v2`) to embed clinical guidelines. |
| `train_and_convert.ipynb` | Jupyter notebook containing the dataset fetching (UCI CDC Health Indicators), SMOTE class balancing, LightGBM training, and ONNX conversion. |
| `diabetes_predictor.onnx` | The compiled binary LightGBM model utilized by the frontend browser. |
| `Dockerfile` | Standardized build steps for production container deployment. |
| `requirements.txt` | Python package dependencies. |

---

## 🛠️ Local Installation & Setup

### 1. Prerequisites

Ensure you have **Python 3.12+** installed.

### 2. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd healthaiot-platform
pip install -r requirements.txt
```

### 3. Environment Variables Config

Create a `.env` file in the root directory to hold your secure cloud credentials:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-supabase-api-key
GEMINI_API_KEY=your-gemini-api-key
```

### 4. Populate the Database

Run the ingestion script to embed and upload the clinical guidelines to your database:

```bash
python seed_db.py
```

### 5. Start the Application

Run the backend server:

```bash
python server.py
```

Open `index.html` in your browser (or use VS Code Live Server) to start using the full-stack system.

---

## 📄 License & Attribution

This project is an open-source development implementation inspired by the architectural concepts introduced in:

> **HealthAIoT: AIoT-driven smart healthcare system for sustainable cloud computing environments** (2025).
