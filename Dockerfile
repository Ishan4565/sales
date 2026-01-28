# 1. Use a standard Python image
FROM python:3.11-slim

# 2. Set the folder inside the cloud
WORKDIR /app

# 3. Install system tools needed for PostgreSQL
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy EVERY file from your GitHub into the cloud folder
COPY . .

# 6. Streamlit needs this port to talk to Render
EXPOSE 8000

# 7. THE CRITICAL LINE: Make sure 'app.py' matches your filename exactly
CMD ["streamlit", "run", "sales_forecast_app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]


