FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Le due differenze chiave:
- `COPY . /app/` invece di `COPY api /app/api` → copia **tutto** il repo incluso `main.py`
- `main:app` e porta `8080` coerente con Render

---

Dopo il commit, aspetta che Render finisca il redeploy (~3 min), poi apri:
```
https://nil-rag-copilot.onrender.com/docs
