FROM python:3.12-slim

WORKDIR /app

# Copy dependency list first (Docker caches this layer if requirements don't change - faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy the actual application code
COPY src/ ./src/
COPY api/ ./api/

# Tell Docker this container listens on port 8000
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]