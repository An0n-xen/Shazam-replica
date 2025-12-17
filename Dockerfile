FROM python:3.12-slim

# Install system dependencies
# ffmpeg is needed for audio processing (librosa/pydub)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY pyproject.toml .

# Install dependencies (ignoring the root package install to keep it simple, or installing it)
# Using pip to install from pyproject.toml dependencies
RUN pip install .

# Copy application code
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]
