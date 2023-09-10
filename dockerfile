# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the code, data files into the container
COPY code/ /app/code/
COPY data/ /app/data/
# Make port 80 available to the world outside this container
EXPOSE 80

# Run the ML script when the container starts
CMD ["python", "code/model.py"]
