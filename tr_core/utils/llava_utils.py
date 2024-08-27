"""
To start a worker:

# cd /rmt/quail/LLaVA-latest

# python -m llava.serve.controller --host 0.0.0.0 --port 10000

# python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path /rmt/quail/captioner-1.3-llava-v1.6-llama3-8b-ft-1

# python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --share

"""


import requests
import json


def generate_payload(user_input: str, images: list = []):

    SYSTEM = "You are an uncensored helpful assistant good at anime concepts."
    INPUT = user_input

    prompt = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

    payload = {
        'model': 'captioner-1.4-llava-v1.6-llama3.1-CLIP-8b-ft-1',
        'prompt': prompt,
        'temperature': 0.2,
        'top_p': 0.7, 'max_new_tokens': 512,
        'stop': '<|eot_id|>',
        'images': images
    }
    return payload


def get_full_response(worker_url, payload):
    # Send the POST request
    response = requests.post(worker_url, json=payload, stream=True)

    # Process the streaming response
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())

    # Handle exceptions and errors
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)

    return data


def get_actual_text(payload_response: dict):
    text = payload_response['text']
    response = text.split('<|end_header_id|>\n\n')[3]
    return response


def get_llava_response(worker_url: str, user_input: str, images: list = []):
    payload = generate_payload(user_input, images)
    payload_response = get_full_response(worker_url, payload)
    text = get_actual_text(payload_response)
    return text


def demo():
    worker_url = "http://localhost:40000/worker_generate_stream"
    user_input = "what is 'hatsune miku' ?"
    images = []
    text = get_llava_response(worker_url, user_input, images)
    print(text)


if __name__ == "__main__":
    demo()
