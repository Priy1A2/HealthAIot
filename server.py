import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

app = Flask(__name__)
# Enable CORS so our local index.html file can send cross-origin requests directly to this server
CORS(app)

# 1. Credentials Setup (Your actual working keys)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Clients Initialization
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception:
    supabase = None

client = genai.Client(api_key=GEMINI_API_KEY)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Knowledge base of clinical guidelines (ADA Standards of Care)
CLINICAL_GUIDELINES = [
    {
        "source_citation": "ADA Standards of Care in Diabetes 2024",
        "risk_factor": "HighBP",
        "clinical_recommendation": "For people with diabetes and hypertension, the ADA recommends a target blood pressure of <130/80 mmHg. Lifestyle therapies include sodium restriction (<2,300 mg/day), weight loss, a DASH-style eating pattern, and increased physical activity."
    },
    {
        "source_citation": "ADA Standards of Care in Diabetes 2024",
        "risk_factor": "BMI",
        "clinical_recommendation": "For overweight or obese adults with type 2 diabetes risk, weight loss of at least 5-7% of initial body weight is highly recommended. This should be combined with intensive lifestyle interventions such as a healthy diet and weekly physical activity."
    },
    {
        "source_citation": "ADA Standards of Care in Diabetes 2024",
        "risk_factor": "HighChol",
        "clinical_recommendation": "Statin therapy is recommended for primary prevention of cardiovascular disease in adults with diabetes aged 40-75 years. Target LDL cholesterol reduction of 50% or more from baseline using moderate- to high-intensity statins."
    },
    {
        "source_citation": "ADA Standards of Care in Diabetes 2024",
        "risk_factor": "PhysActivity",
        "clinical_recommendation": "Adults with diabetes or prediabetes should engage in 150 minutes or more of moderate-to-vigorous intensity aerobic physical activity per week, spread over at least 3 days per week, with no more than 2 consecutive days without activity."
    },
    {
        "source_citation": "ADA Standards of Care in Diabetes 2024",
        "risk_factor": "Smoker",
        "clinical_recommendation": "All patients are advised not to use cigarettes, other tobacco products, or e-cigarettes. Incorporate smoking cessation counseling and pharmacological treatment options as standard preventive care."
    },
    {
        "source_citation": "ADA Standards of Care in Diabetes 2024",
        "risk_factor": "GenHlth",
        "clinical_recommendation": "Annual comprehensive assessments of microvascular and macrovascular complications are recommended. Self-monitored blood glucose tracking and self-reported health ratings help clinicians tailor pharmacotherapy."
    }
]

GUIDELINE_EMBEDDINGS = [
    embedding_model.encode(item["clinical_recommendation"]) for item in CLINICAL_GUIDELINES
]

def search_clinical_guidelines(query_embedding, match_count=2):
    # 1. Try Supabase pgvector RPC first
    if supabase:
        try:
            rpc_response = supabase.rpc(
                "match_clinical_knowledge", 
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.1, 
                    "match_count": match_count        
                }
            ).execute()
            if rpc_response.data and len(rpc_response.data) > 0:
                return rpc_response.data
        except Exception as err:
            print(f"[Supabase RPC Notice] Remote vector search unavailable ({err}). Using local vector similarity.")

    # 2. Local semantic cosine similarity search fallback
    q_vec = np.array(query_embedding)
    scores = []
    for g_vec in GUIDELINE_EMBEDDINGS:
        cos_sim = float(np.dot(q_vec, g_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(g_vec)))
        scores.append(cos_sim)
    
    top_indices = np.argsort(scores)[::-1][:match_count]
    return [CLINICAL_GUIDELINES[i] for i in top_indices]

SYSTEM_INSTRUCTION = """
You are a highly secure, professional clinical interpreter for a smart healthcare system. 
Your task is to explain a patient's numerical diabetes risk metrics using only authorized clinical guidelines.

Strict Compliance Rules:
1. Do NOT diagnose the patient. Use words like 'Your risk profile suggests' or 'Guidelines advise'.
2. You must rely ONLY on the provided Clinical Context below. Do not assume or introduce medical knowledge outside this context.
3. Explicitly cite your clinical sources (e.g., 'According to the ADA Standards of Care 2024...').
4. Keep the explanation structured, patient-friendly, and actionable, emphasizing lifestyle modifications.
"""

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

@app.route('/api/explain', methods=['POST'])
def explain_risk():
    try:
        data = request.json
        risk_score = data.get("risk_score")
        features = data.get("features")

        # 1. Assemble Semantic Query based on patient risk indicators
        high_risks = []
        if features.get("HighBP") == 1: high_risks.append("high blood pressure")
        if features.get("BMI", 0) >= 25.0: high_risks.append("overweight body mass index BMI")
        if features.get("HighChol") == 1: high_risks.append("high cholesterol")
        if features.get("PhysActivity") == 0: high_risks.append("low physical activity sedentary lifestyle")

        query_text = "preventive advice for " + " and ".join(high_risks) if high_risks else "general diabetes preventative care lifestyle guidelines"

        # 2. Generate Vector Embeddings and Perform Semantic Retrieval
        query_embedding = embedding_model.encode(query_text).tolist()
        retrieved_guidelines = search_clinical_guidelines(query_embedding, match_count=2)

        # 3. Format Context
        context_block = ""
        for idx, item in enumerate(retrieved_guidelines):
            context_block += f"\nSource [{idx+1}]: {item['source_citation']}\n"
            context_block += f"Category: {item['risk_factor']}\n"
            context_block += f"Recommendation: {item['clinical_recommendation']}\n"

        prompt = f"""
        Patient Risk Profile:
        - Calculated Diabetes Risk Score: {risk_score * 100:.2f}%
        - Key Features: BMI: {features.get('BMI')}, High Blood Pressure: {'Yes' if features.get('HighBP') == 1 else 'No'}, High Cholesterol: {'Yes' if features.get('HighChol') == 1 else 'No'}, Physical Activity: {'Active' if features.get('PhysActivity') == 1 else 'Inactive'}

        Clinical Context Guidelines:
        {context_block}

        Based on the profile and context, write a safe, compliant clinical explanation and preventive action plan for the patient.
        """

        # 4. Generate Content via Gemini 2.5 Flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2
            )
        )

        return jsonify({"report": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)