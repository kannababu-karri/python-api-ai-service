from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os

log_dir = "/opt/apps/ai-service/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "python-aiagent-api.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("AI API service started")

# -----------------------------
# App Init
# -----------------------------
app = FastAPI()

# -----------------------------
# Upload Size Limit Middleware
# -----------------------------
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):

        if request.headers.get("content-length"):
            if int(request.headers["content-length"]) > 50 * 1024 * 1024:
                return Response("File too large (Max 50MB)", status_code=413)

        return await call_next(request)


app.add_middleware(LimitUploadSizeMiddleware)

logging.info("Initializing OpenAI LLM")
# -----------------------------
# AI Model (Loaded Once)
# -----------------------------
#llm = OllamaLLM(
#    model="llama3",
#    temperature=0.2
#)

llm = ChatOpenAI(
    model="gpt-4o-mini",   # Fast + cheap + good quality
    temperature=0.2,
    max_tokens=800
)

logging.info("Initializing OpenAI LLM-Ending")


# -----------------------------
# Text Splitter
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=400
)


# -----------------------------
# Thread Pool
# -----------------------------
executor = ThreadPoolExecutor(max_workers=4)


# -----------------------------
# Analyze One Chunk
# -----------------------------
async def analyze_chunk(i, chunk):

    loop = asyncio.get_event_loop()

    prompt = f"""
You are a pharmaceutical GMP compliance auditor.

Analyze this document section and identify:

- Violations
- Missing procedures
- Regulatory risks
- Deviations

Return concise bullet points.

Section {i + 1}:
{chunk}
"""

    result = await loop.run_in_executor(
        executor,
        llm.invoke,
        prompt
    )

    return f"### Section {i + 1}\n{result}"


# -----------------------------
# Main API (For Spring Boot)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "UP"}

@app.post("/analyze")
async def analyze_pdf(
        file: UploadFile = File(...),
        batchId: str = Form(...)
):
    try:
        logging.info("Reading PDF file: %s", file.filename)
        # -----------------------------
        # Read PDF
        # -----------------------------
        reader = PdfReader(file.file)

        full_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"


        if not full_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "No readable text found in PDF"}
            )


        # -----------------------------
        # Split into Chunks
        # -----------------------------
        chunks = text_splitter.split_text(full_text)

        if not chunks:
            return JSONResponse(
                status_code=400,
                content={"error": "Text chunking failed"}
            )


        # -----------------------------
        # Run AI in Parallel
        # -----------------------------
        tasks = [
            analyze_chunk(i, chunk)
            for i, chunk in enumerate(chunks)
        ]

        findings = await asyncio.gather(*tasks)

        final_report = "\n\n".join(findings)


        # -----------------------------
        # Response to Spring Boot
        # -----------------------------
        return JSONResponse({

            "batchId": batchId,
            "total_chunks": len(chunks),
            "report": final_report

        })


    except Exception as e:
        logging.error("AI processing failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "error": "AI processing failed",
                "details": str(e)
            }
        )