FROM 3.10-bookworm
COPY ./requirements.txt .
RUN pip install -r ./requirements.txt
COPY . .