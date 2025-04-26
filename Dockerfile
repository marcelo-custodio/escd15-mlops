# Start from the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install supervisor to manage multiple processes
RUN apt update && apt -y upgrade && apt install -y supervisor

# Copy the app code
COPY . .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Get initial model
RUN

# Expose the port MLFlow will run on
EXPOSE 5000
# Expose the port FastAPI will run on
EXPOSE 8000

# Start the app with uvicorn
CMD ["supervisord", "-n"]