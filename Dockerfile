# 1. Use a lightweight Python image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies for PostgreSQL (psycopg2)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code AND the large .pkl files into the container
COPY . .

# 6. Tell Render which port your app uses (usually 8000 or 5000)
EXPOSE 8000

# 7. Start the app (adjust 'app:app' if your file is named differently)
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]