FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- THE MAGIC STEP ---
# Copy the builder script and RUN it. 
# This bakes the 4GB file into the Docker Image itself.
COPY builder.py .
RUN python builder.py
# ----------------------

COPY handler.py .

# Make sure your handler code knows to look in "/model" 
# instead of trying to download it again!
CMD [ "python", "-u", "/app/handler.py" ]
