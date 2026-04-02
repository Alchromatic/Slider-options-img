### Stage 1: build the 'primitive' Go binary
FROM golang:1.24-bookworm AS go-builder
RUN git clone https://github.com/fogleman/primitive.git /tmp/primitive && \
    cd /tmp/primitive && \
    go mod init github.com/fogleman/primitive && \
    echo 'require golang.org/x/image v0.23.0' >> go.mod && \
    echo 'require github.com/fogleman/gg v1.3.0' >> go.mod && \
    echo 'require github.com/golang/freetype v0.0.0-20170609003504-e2365dfdc4a0' >> go.mod && \
    echo 'require github.com/nfnt/resize v0.0.0-20180221191011-83c6a9932646' >> go.mod && \
    go mod download && \
    CGO_ENABLED=0 go build -o /usr/local/bin/primitive .

### Stage 2: final image (no Go toolchain)
FROM python:3.11-slim

# Copy pre-built primitive binary
COPY --from=go-builder /usr/local/bin/primitive /usr/local/bin/primitive

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
