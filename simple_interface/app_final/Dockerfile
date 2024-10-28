# Use an official Python runtime as a base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt first to leverage Docker's caching
COPY requirements.txt ./

# Install any necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app_copy.py", "--server.port=8501", "--server.enableCORS=false"]
