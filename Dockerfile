# Use the official Python image
FROM python:3.10.5

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . .

# Install FastAPI and Uvicorn
RUN pip install \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    numpy==1.26.3 \
    pandas==2.2.1 \
    scikit-learn==1.3.2 \
    FLAML==2.1.1 \
    matplotlib==3.8.2 \
    colored==2.2.4 \
    plotly==5.18.0 \
    xgboost==2.0.3 \
    supabase==2.4.3 \
    dill==0.3.8 \
    email_validator==2.2.0

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]