FROM python:3.11-slim

ARG USERNAME=ltmuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV PYTHONUNBUFFERED=1
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /app
RUN chown -R $USERNAME:$USERNAME /app

USER $USERNAME

COPY --chown=$USERNAME:$USERNAME requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

COPY --chown=$USERNAME:$USERNAME app/ ./app/

RUN mkdir -p /app/logs

ENV PYTHONUNBUFFERED=1
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333
ENV EMBEDDING_MODEL=intfloat/multilingual-e5-large

EXPOSE 8006

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006"] 