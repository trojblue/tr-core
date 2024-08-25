import ast
from http import client
import re

import os
import tempfile
from gradio_client import Client, handle_file
from PIL import Image

import unibox as ub

import logging
logger = logging.getLogger(__name__)

class GradioApiCaller:
    def __init__(self):
        self.client_url = None
        self.client = None
        self.temp_files = []

    def _is_url(self, value: str) -> bool:
        """
        Checks if the given string is a URL.
        """
        url_pattern = re.compile(
            r'^(https?|ftp)://[^\s/$.?#].[^\s]*$', re.IGNORECASE)
        return re.match(url_pattern, value) is not None

    def _prepare_args(self, args):
        """
        Prepares the arguments, handling special cases like FILE: for file inputs.

        Args:
            args (dict): The arguments to be passed to the Gradio API.

        Returns:
            dict: The prepared arguments.
        """
        prepared_args = {}
        for key, value in args.items():
            if isinstance(value, str):
                if value.startswith("FILE:"):
                    # Handle file inputs by wrapping with handle_file
                    file_path = value[len("FILE:"):]
                    prepared_args[key] = handle_file(file_path)
                elif self._is_url(value):
                    # If the value is a URL but doesn't start with FILE:, raise a warning
                    logger.warning(
                        f"Warning: The argument '{key}' appears to be a URL but doesn't start with 'FILE:'. "
                        f"Did you forget to add 'FILE:' in front of it?"
                    )
                    prepared_args[key] = value
                else:
                    prepared_args[key] = value
            else:
                prepared_args[key] = value
        return prepared_args

    def _save_non_serializable(self, key, value):
        """
        Saves non-JSON-serializable objects (like PIL images or DataFrames) to a temporary file.

        Args:
            key (str): The key of the argument.
            value: The non-serializable object.

        Returns:
            str: The file path wrapped with "FILE:" prefix.
        """
        if isinstance(value, Image.Image):
            # Determine the appropriate file extension based on the image format
            extension = value.format.lower() if value.format else 'png'
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}') as temp_file:
                # Keep track of the temp file for cleanup
                self.temp_files.append(temp_file.name)
                value.save(temp_file.name)
                temp_file.flush()  # Ensure all data is written to disk
            return f"FILE:{temp_file.name}"
        else:
            # Handle other types or raise an error
            raise ValueError(
                f"Unsupported type for key '{key}': {type(value)}")

    def _cleanup_temp_files(self):
        """
        Cleans up temporary files created during the API call.
        """
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except OSError as e:
                print(f"Error: {temp_file} : {e.strerror}")
        self.temp_files = []  # Reset the list after cleanup

    def _call_gradio_api(self, client_url: str, default_args: dict, override_args: dict) -> dict:
        # Ensure no extra arguments are provided
        extra_args = set(override_args) - set(default_args)
        if extra_args:
            raise ValueError(
                f"Extra arguments provided that are not in the default arguments: {extra_args}")

        # Merge the default arguments with the overrides
        merged_args = {**default_args, **override_args}

        # Handle non-JSON-serializable objects
        for key, value in merged_args.items():
            try:
                # Test if the value is JSON-serializable
                import json
                json.dumps(value)
            except (TypeError, ValueError):
                # If not serializable, save to temp file
                merged_args[key] = self._save_non_serializable(key, value)

        # Prepare the arguments, handling FILE: inputs
        prepared_args = self._prepare_args(merged_args)

        # Initialize a new client only if the client_url has changed
        if self.client_url != client_url:
            self.client_url = client_url
            self.client = Client(client_url)

        # Call the API
        result = self.client.predict(**prepared_args)
        return result

    def call_gradio_api(self, client_url: str, default_args: dict, override_args: dict) -> dict:
        """
        Calls the Gradio API with the provided arguments, allowing overrides but not extra arguments.

        Args:
            client_url (str): The URL of the Gradio client.
            default_args (dict): The default arguments to be used if not overridden.
            override_args (dict): The arguments to override the defaults.

        Returns:
            dict: The result from the Gradio API call.
        """
        try:
            return self._call_gradio_api(client_url, default_args, override_args)

        finally:
            # Clean up temporary files after the API call
            self._cleanup_temp_files()

    def __call__(self, client_url: str, default_args: dict, override_args: dict) -> dict:
        """
        Allows the class instance to be called like a function.
        """
        return self.call_gradio_api(client_url, default_args, override_args)

class GradioCodeStrParser:
    def __init__(self):
        self.client_url = None
        self.predict_args_str = None
        self.predict_args = {}
        self.parse_file = False

    def extract_client_url(self, code_str):
        """
        Extracts the client URL from the code string.
        """
        client_url_match = re.search(r'Client\("([^"]+)"\)', code_str)
        if client_url_match:
            self.client_url = client_url_match.group(1)
        else:
            raise ValueError("Client URL not found in the provided code.")

    def extract_predict_args_str(self, code_str):
        """
        Extracts the text inside the client.predict(...) call.
        """
        predict_args_match = re.search(
            r'client\.predict\((.*)\)', code_str, re.DOTALL)
        if predict_args_match:
            self.predict_args_str = predict_args_match.group(1).strip()

            # Handle nested structures and capture until the last closing parenthesis
            open_parens = 0
            correct_end = len(self.predict_args_str)

            for i, char in enumerate(self.predict_args_str):
                if char == '(':
                    open_parens += 1
                elif char == ')':
                    if open_parens == 0:
                        correct_end = i
                        break
                    else:
                        open_parens -= 1

            self.predict_args_str = self.predict_args_str[:correct_end].strip()
        else:
            raise ValueError(
                "Predict arguments not found in the provided code.")

    def parse_predict_args(self):
        """
        Parses the predict arguments string into a dictionary.
        """
        if not self.predict_args_str:
            raise ValueError("No predict arguments string to parse.")

        # Split the string by commas that are followed by newlines and spaces
        args_list = [arg.strip() for arg in self.predict_args_str.split(',\n')]

        # Parse each key-value pair into the dictionary
        for arg in args_list:
            key, value = arg.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Check if the value is a handle_file(...) call
            handle_file_match = re.match(r'handle_file\((.*)\)', value)
            if self.parse_file and handle_file_match:
                # Extract the content inside handle_file(...) and remove any wrapping quotes
                file_content = handle_file_match.group(1).strip().strip("'").strip('"')
                self.predict_args[key] = f"FILE:{file_content}"
            else:
                # Convert the value to the appropriate Python type
                try:
                    self.predict_args[key] = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    self.predict_args[key] = value

    def __call__(self, code_str, parse_file=False):
        """
        Allows the class instance to be called like a function.

        Args:
            code_str (str): The code string to parse.
            parse_file (bool): If True, parse the content inside handle_file(...).
                               If False, leave handle_file(...) as is.
        """
        self.parse_file = parse_file
        self.extract_client_url(code_str)
        self.extract_predict_args_str(code_str)
        self.parse_predict_args()

        if not self.client_url:
            raise ValueError("Client URL not found in the provided code.")

        return self.client_url, self.predict_args


def sample_parse_gradio_code_str():

    code_str = """from gradio_client import Client

    client = Client("https://flag-celebration-manually-pan.trycloudflare.com/")
    result = client.predict(
            tags_front="(best quality:1.8)",
            user_input="frederica bernkastel, 1girl",
            tags_back="absurdres, best [quality], 2020s",
            num_images=4,
            seed=-1,
            upscale_prompt=False,
            api_name="/gradio_inference_1"
    )
    print(result)
    """

    # Create an instance of the parser and parse the code string
    parser = GradioCodeStrParser()
    client_url, predict_args = parser(code_str)

    print(client_url)
    print(predict_args)


# Initialize a GradioApiCaller instance when the module is loaded
gradio_caller = GradioApiCaller()
gradio_parser = GradioCodeStrParser()