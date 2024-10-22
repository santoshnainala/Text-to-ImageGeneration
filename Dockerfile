# Use a more complete Python image with Debian-based utilities
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies, including Git
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && git --version

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "ImageGeneration:app", "--host", "0.0.0.0", "--port", "8000"]
