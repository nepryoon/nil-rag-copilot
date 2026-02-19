import io, os, time, uuid, logging
import numpy as np
import pdfplumber
from groq import Groq
from openai import OpenAI
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NIL RAG Copilot API", version="0.5.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"])

class IngestResponse(BaseModel):
    status: str; session_id: str; word_count: int; chunk_count: int; message: str

class ChatRequest(BaseModel):
    session_id: str; question: str

class Citation(BaseModel):
    chunk_id: int; text_snippet: str; score: float

class ChatResponse(BaseModel):
    answer: str; citations: List[Citation]; retrieval_latency_ms: float

class EvalRequest(BaseModel):
    session_id: str

class MetricResult(BaseModel):
    name: str; score: float; description: str

class EvalResponse(BaseModel):
    session_id: str; metrics: List[MetricResult]
    test_questions: List[str]; answers: List[str]

_sessions: Dict[str, Any] = {}
MAX_WORDS, CHUNK_SIZE, OVERLAP, TOP_K = 5000, 200, 40, 4

def get_openai():
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

def embed(texts: List[str]) -> np.ndarray:
    response = get_openai().embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    vecs = np.array([d.embedding for d in response.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)

def extract_text(b: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t: parts.append(t.strip())
    return "\n\n".join(parts)

def validate_words(text: str) -> int:
    wc = len(text.split())
    if wc > MAX_WORDS:
        raise ValueError(f"Il documento contiene {wc} parole. La demo accetta un massimo di {MAX_WORDS} parole.")
    return wc

def make_chunks(text: str) -> List[str]:
    words = text.split(); chunks = []; start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words): break
        start += CHUNK_SIZE - OVERLAP
    return chunks

def retrieve(query: str, embeddings: np.ndarray, chunks: List[str]):
    q = embed([query])[0]
    scores = embeddings @ q
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    return [(int(i), chunks[i], float(scores[i])) for i in top_idx]

def llm(messages, max_tokens=600, temperature=0.1):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    ).choices[0].message.content

@app.get("/health")
def health(): return {"status": "ok", "version": "0.5.0"}

@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    content = await file.read()
    try: text = extract_text(content)
    except Exception as e: raise HTTPException(422, f"Cannot read PDF: {e}")
    if not text.strip(): raise HTTPException(422, "PDF has no extractable text.")
    try: wc = validate_words(text)
    except ValueError as e: raise HTTPException(422, str(e))
    chunks = make_chunks(text)
    embeddings = embed(chunks)
    sid = str(uuid.uuid4())
    _sessions[sid] = {"embeddings": embeddings, "chunks": chunks, "word_count": wc}
    logger.info(f"Ingest OK session={sid} words={wc} chunks={len(chunks)}")
    return IngestResponse(status="ok", session_id=sid, word_count=wc,
        chunk_count=len(chunks),
        message=f"Indexing complete. {wc} words, {len(chunks)} chunks indexed.")

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.session_id not in _sessions: raise HTTPException(404, "Session not found.")
    s = _sessions[req.session_id]
    t0 = time.perf_counter()
    results = retrieve(req.question, s["embeddings"], s["chunks"])
    latency = (time.perf_counter() - t0) * 1000
    context = "\n\n".join(f"[Chunk {i}]: {t}" for i, t, _ in results)
    answer = llm([
        {"role": "system", "content": "Answer ONLY from context. If not found say: 'I could not find this in the document.' Cite as [Chunk N]."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
    ])
    return ChatResponse(answer=answer,
        citations=[Citation(chunk_id=i, text_snippet=t[:150]+"…", score=sc) for i, t, sc in results],
        retrieval_latency_ms=round(latency, 2))

@app.post("/api/v1/eval", response_model=EvalResponse)
async def run_eval(req: EvalRequest):
    if req.session_id not in _sessions: raise HTTPException(404, "Session not found.")
    s = _sessions[req.session_id]
    chunks, embeddings = s["chunks"], s["embeddings"]
    questions = []
    step = max(1, len(chunks) // 5)
    for i in range(0, min(5 * step, len(chunks)), step):
        sentence = chunks[i].split(".")[0].strip()
        if len(sentence) > 15:
            questions.append(f"What does this describe: '{sentence[:80]}'?")
    if not questions: raise HTTPException(422, "Cannot generate test questions.")
    answers = []
    for q in questions:
        r = retrieve(q, embeddings, chunks)
        ctx = "\n\n".join(f"[Chunk {i}]: {t}" for i, t, _ in r)
        answers.append(llm([
            {"role": "system", "content": "Answer only from context."},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}
        ], max_tokens=200, temperature=0.0))
    rp = float(np.mean([retrieve(q, embeddings, chunks)[0][2] for q in questions]))
    q_emb = embed(questions)
    a_emb = embed(answers)
    ar = float(np.mean(np.sum(q_emb * a_emb, axis=1)))
    used = {i for q in questions for i, _, _ in retrieve(q, embeddings, chunks)}
    cc = len(used) / len(chunks) if chunks else 0.0
    return EvalResponse(session_id=req.session_id, test_questions=questions, answers=answers,
        metrics=[
            MetricResult(name="Retrieval Precision", score=round(rp, 3), description="Avg top-1 cosine score — OpenAI text-embedding-3-small (0–1)"),
            MetricResult(name="Answer Relevance", score=round(ar, 3), description="Avg cosine similarity question/answer (0–1)"),
            MetricResult(name="Context Coverage", score=round(cc, 3), description="Fraction of chunks used in retrievals (0–1)"),
        ])
