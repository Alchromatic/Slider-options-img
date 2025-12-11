FROM python:3.11-slim

# Install Go
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz && \
    rm go1.21.5.linux-amd64.tar.gz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/usr/local/go/bin:/root/go/bin

# Install primitive
RUN go install github.com/fogleman/primitive@latest

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Run the app
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
