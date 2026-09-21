FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gymnasium stable-baselines3 imageio pillow seaborn

COPY . .

# Run short benchmark demonstration on launch
CMD ["python", "scripts/compare_controllers.py", "--episodes", "10", "--mock-train"]
