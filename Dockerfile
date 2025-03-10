# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files
COPY train.py app.py model.pkl requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
