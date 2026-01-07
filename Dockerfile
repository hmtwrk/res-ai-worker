FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

WORKDIR /app

# Define the argument
ARG HF_TOKEN

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY builder.py .
# Pass the ARG into the ENV so the python script can see it
RUN HF_TOKEN=${HF_TOKEN} python builder.py

COPY handler.py .
CMD [ "python", "-u", "/app/handler.py" ]
