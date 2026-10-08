import os
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from langchain_huggingface import HuggingFaceEmbeddings
import io
import numpy as np
from fastapi import File, UploadFile
from fastapi.concurrency import run_in_threadpool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import uuid
import boto3

app = FastAPI()
context = CryptContext(schemes=["sha512_crypt"])
bearer = HTTPBearer()


S3_BUCKET = os.environ["S3_BUCKET"]
s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "eu-west-2"))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 10  

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = {}  # clientID -> list of {"filename", "text", "embedding"}

SECRET_KEY = os.environ.get("JWT_SECRET", "change-me-in-production")
ALGORITHM = "HS256"
TOKEN_LIFETIME = timedelta(hours=1)

clients = {}  

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

class QueryRequest(BaseModel):
       message: str = Field(min_length=1)

class ClientRequest(BaseModel):
    clientEmail : EmailStr
    clientPassword : str = Field(min_length=8, max_length=128)


def create_token(email: str, client_id: int) -> str:
    payload = {
        "sub": email,
        "clientID": client_id,
        "exp": datetime.now(timezone.utc) + TOKEN_LIFETIME,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_client(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired, please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    client = clients.get(payload["sub"])
    if client is None:
        raise HTTPException(status_code=401, detail="Client no longer exists")
    return client

@app.get("/")
async def home():
    return {"message" : "home"}

@app.post("/sign-up", status_code=201)
def sign_up(req: ClientRequest):
    email = req.clientEmail.lower()
    if email in clients:
        raise HTTPException(status_code=409, detail="Email already registered")

    client_id = len(clients) + 1
    clients[email] = {
        "clientID": client_id,
        "passwordHash": context.hash(req.clientPassword),
    }
    return {"message": "Account created", "clientID": client_id}

@app.post("/login")
def login(req: ClientRequest):
    email = req.clientEmail.lower()
    client = clients.get(email)

    if client is None or not context.verify(req.clientPassword, client["passwordHash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "access_token": create_token(email, client["clientID"]),
        "token_type": "bearer",
    }

@app.post("/query")
def process_query(req: QueryRequest, client: dict = Depends(get_current_client)):
    client_docs = documents.get(client["clientID"], [])
    if not client_docs:
        raise HTTPException(status_code=404, detail="No documents uploaded yet")

    query_emb = np.array(hf.embed_query(req.message))
    doc_embs = np.array([d["embedding"] for d in client_docs])

    scores = doc_embs @ query_emb          # similarity score for every chunk
    top = np.argsort(scores)[::-1][:3]     # indexes of the 3 best matches

    return {
        "results": [
            {
                "filename": client_docs[i]["filename"],
                "text": client_docs[i]["text"],
                "score": float(scores[i]),
            }
            for i in top
        ]
    }

def extract_and_embed(contents: bytes) -> tuple[list[str], list[list[float]]]:
    reader = PdfReader(io.BytesIO(contents))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = splitter.split_text(text)
    embeddings = hf.embed_documents(chunks) if chunks else []
    return chunks, embeddings

@app.post("/upload", status_code=201)
async def process_pdf_upload(
    file: UploadFile = File(...),
    client: dict = Depends(get_current_client),
):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")
    if not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        chunks, embeddings = await run_in_threadpool(extract_and_embed, contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read PDF")

    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in PDF (it may be scanned images)")

    s3_key = f"clients/{client['clientID']}/{uuid.uuid4()}.pdf"
    try:
        await run_in_threadpool(
            s3.put_object,
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=contents,
            ContentType="application/pdf",
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Could not store file")

    client_docs = documents.setdefault(client["clientID"], [])
    for chunk, emb in zip(chunks, embeddings):
        client_docs.append({
            "filename": file.filename,
            "s3_key": s3_key,
            "text": chunk,
            "embedding": emb,
        })

    return {"message": "PDF uploaded", "filename": file.filename, "chunks": len(chunks)}