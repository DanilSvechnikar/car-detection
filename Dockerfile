FROM python:3.12-slim-bullseye

WORKDIR /car-detection

USER root
EXPOSE 8888

# Install linux packages
RUN apt update && apt upgrade -y \
    && apt install libgl1 libglib2.0-0 -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
# Run 'make poetry-export' on host machine to get requirements.txt If it doesn't exist
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy sources
COPY car_detection/ ./car_detection
COPY models/ ./models
COPY data/demo_data/ ./data/demo_data
COPY templates ./templates
COPY .env main.py api_test.py ./

# Cleaning
RUN rm requirements.txt
RUN pip3 cache purge

CMD ["python", "main.py"]
#CMD ["python", "-f", "/dev/null"]
