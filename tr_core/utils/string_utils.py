import ast
import re


sample_code_str = """
from gradio_client import Client

client = Client("https://flag-celebration-manually-pan.trycloudflare.com/")
result = client.predict(
        character="Hello!!",
        general="Hello!!",
        rating="general",
        batch_size=1,
        api_name="/demo_gpt2_prompt_upscaler"
)
print(result)
"""

def parse_gradio_code_to_params(code_str: str) -> tuple:
    """
    Parses a Gradio inference code string to extract the client URL and prediction arguments.

    Args:
        code_str (str): The Gradio inference code as a string.

    Returns:
        tuple: A tuple containing the client URL (str) and prediction arguments (dict).
    
    >>> code_str = sample_code_str
    >>> client_url, predict_args = parse_gradio_code_to_params(code_str)
    """

    # Extract the client URL
    client_url_match = re.search(r'Client\("([^"]+)"\)', code_str)
    if not client_url_match:
        raise ValueError("Client URL not found in the provided code.")
    client_url = client_url_match.group(1)
    
    # Extract the predict arguments by searching for the client.predict call
    predict_call_match = re.search(r'client\.predict\((.*?)\)', code_str, re.DOTALL)
    if not predict_call_match:
        raise ValueError("Predict arguments not found in the provided code.")
    
    # The predict args are in a comma-separated string form
    predict_args_str = predict_call_match.group(1).strip()

    # Convert the arguments string to a dictionary
    predict_args = {}
    for arg in predict_args_str.split(',\n'):
        key, value = arg.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Convert the value to the appropriate Python type
        predict_args[key] = ast.literal_eval(value)
    
    return client_url, predict_args