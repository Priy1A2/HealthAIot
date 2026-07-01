import os

import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# 1. Initialize connection to Supabase using your project credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

print("Step 1: Downloading open-source embedding model locally (all-MiniLM-L6-v2)...")
# This model outputs exactly 384 dimensions, matching our public.clinical_knowledge database schema
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded successfully.")

print("\nStep 2: Preparing clinical knowledge guidelines...")
clinical_guidelines = [
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

print("\nStep 3: Generating vector embeddings and inserting data into Supabase...")
for item in clinical_guidelines:
    text_to_embed = item["clinical_recommendation"]
    
    # Generate embedding and convert output numpy array to a python list of floats
    embedding_vector = model.encode(text_to_embed).tolist()
    
    # Map to schema payload
    payload = {
        "source_citation": item["source_citation"],
        "risk_factor": item["risk_factor"],
        "clinical_recommendation": item["clinical_recommendation"],
        "embedding": embedding_vector
    }
    
    # Insert data via Supabase PostgREST Client
    response = supabase.table("clinical_knowledge").insert(payload).execute()
    print(f"-> Successfully inserted guideline for risk factor: {item['risk_factor']}")

print("\nAll clinical guidelines have been embedded and securely uploaded to Supabase.")