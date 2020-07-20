FROM pytorch/pytorch:latest

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CMD python home/repo/nextitnet-master/app/nextitrec.py

#CMD ls home/repo/nextitnet-master/Data/Sessiond