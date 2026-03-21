FROM python:3.11-slim

# Install Go 1.24
RUN apt-get update && \
    apt-get install -y wget git && \
    wget https://go.dev/dl/go1.24.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.24.0.linux-amd64.tar.gz && \
    rm go1.24.0.linux-amd64.tar.gz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/usr/local/go/bin:/root/go/bin

# Install primitive
ENV GOFLAGS=-mod=mod
RUN git clone https://github.com/fogleman/primitive.git /tmp/primitive && \
    cd /tmp/primitive && \
    go mod init github.com/fogleman/primitive && \
    go get github.com/fogleman/gg && \
    go get github.com/golang/freetype/raster && \
    go get github.com/nfnt/resize && \
    go get golang.org/x/image@v0.23.0 && \
    go install . && \
    rm -rf /tmp/primitive

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Run the app
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}