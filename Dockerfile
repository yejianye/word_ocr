# Use Python 3.10 slim as base image
FROM python:3.10-slim as builder

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

## Install binary dependencies
#COPY packages.txt .
#RUN apt-get update && \
#    xargs apt-get install -y < packages.txt && \
#    rm -rf /var/lib/apt/lists/* && \
#    rm packages.txt

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"] 
