# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install Python dependencies inside the Docker container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code from the current directory to the container
COPY . .

# Specify the default command to run your training script
CMD ["python", "Train.py"]
