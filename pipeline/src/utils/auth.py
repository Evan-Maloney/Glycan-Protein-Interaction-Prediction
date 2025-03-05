import os
from huggingface_hub import login

def authenticate_huggingface():
    token = os.environ.get('HF_TOKEN')
    if token is None:
        raise ValueError("HF_TOKEN environment variable not found")
    login(token=token)