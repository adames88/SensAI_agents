# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install sqlite3 version >= 3.35.0
RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev

# Set the working directory
WORKDIR /SensAI_agents

# Copy the current directory contents into the container at /app
COPY . /SensAI_agents

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run your app
CMD ["streamlit", "run", "app.py"]
