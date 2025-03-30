from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import os
import aiofiles
import zipfile
import shutil
import tempfile
import pypdf
import pandas as pd

# /// script
# requires-python = ">=3.13"
# dependencies = ["fastapi","uvicorn","requests","python-multipart","aiofiles","pypdf","pandas"]      
# ///

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

URL = "http://aiproxy.sanand.workers.dev/openai/v1"
APIPROXY_TOKEN = os.environ.get("APIPROXY_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {APIPROXY_TOKEN}",
}

def extract_text_from_pdf(file_path: str) -> str:
    with open(file_path, "rb") as f:
        pdf_reader = pypdf.PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)

def extract_text_from_csv(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        return df.to_string()
    except Exception:
        return "Error reading CSV file."

def extract_text_from_markdown(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def process_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".csv"):
        return extract_text_from_csv(file_path)
    elif file_path.endswith(".md") or file_path.endswith(".txt"):
        return extract_text_from_markdown(file_path)
    return ""

async def save_uploaded_file(uploaded_file: UploadFile) -> str:
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.filename)
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await uploaded_file.read())
    return file_path, temp_dir

@app.get("/")
def home():
    return "Hello, World!"
    
@app.post("/api")
async def llm_assignment(question: str = Form(...), file: UploadFile = File(None)):
    try:
        extracted_text = ""
        temp_dir = None

        if file:
            file_path, temp_dir = await save_uploaded_file(file)
            
            if zipfile.is_zipfile(file_path):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                for root, _, files in os.walk(temp_dir):
                    for extracted_file in files:
                        extracted_text += process_file(os.path.join(root, extracted_file)) + "\n"
            else:
                extracted_text = process_file(file_path)

        system_prompt = (
            "You are a helpful assistant helping a student with their assignment. "
            "Review the provided content before answering. Provide only the final answer without any additional text or information. "
            "For example, If the question is 'What is the capital of France?' The answer should be 'Paris'. (Strictly follow this answer format for all questions.)"
        )
        
        if extracted_text.strip():
            system_prompt += f"\nHere is the extracted file content:\n{extracted_text}"
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
        }

        response = requests.post(f"{URL}/chat/completions", headers={**HEADERS, "Content-Type": "application/json"}, json=data)
        response_data = response.json()

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response_data)

        final_answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        return JSONResponse(content={"answer": final_answer})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
