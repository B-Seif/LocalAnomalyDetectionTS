FROM registry.gitlab.hpi.de/akita/i/python3-base:latest
ENV ALGORITHM_MAIN="/app/TDP_AD.py"
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY TDP_AD.py /app/
