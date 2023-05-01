# Use an official Python runtime as a parent image
FROM python:3.7-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the streamlit app file into the container at /app
COPY stream1.py /app

# Copy the weights_best.h5 file into the container at /app
COPY weights_best.h5 /app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit when the container launches
CMD ["streamlit", "run", "stream1.py"]