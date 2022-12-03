FROM public.ecr.aws/lambda/python:3.8

WORKDIR /var/task

COPY requirements.txt requirements.txt

RUN  ["python3", "-m" , "pip", "install", "--upgrade", "pip"]

RUN ["pip", "install", "-r", "requirements.txt", "--no-cache-dir"]

COPY . .

CMD ["test_defend.lambda_handler"]