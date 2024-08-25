import ast
import re
from gradio_client import Client


class GradioApiCaller:
    def __init__(self):
        self.client_url = None
        self.client = None

    def call_gradio_api(self, client_url: str, default_args: dict, override_args: dict) -> dict:
        """
        Calls the Gradio API with the provided arguments, allowing overrides but not extra arguments.

        Args:
            client_url (str): The URL of the Gradio client.
                e.g., "https://flag-celebration-manually-pan.trycloudflare.com/"
            
            default_args (dict): The default arguments to be used if not overridden.
                e.g., {"user_input": "frederica bernkastel, 1girl", "num_images": 4, "seed": -1}
            
            override_args (dict): The arguments to override the defaults.
                e.g., {"user_input": "ootori emu, 1girl"}

        Returns:
            dict: The result from the Gradio API call.
        """
        # Ensure no extra arguments are provided
        extra_args = set(override_args) - set(default_args)
        if extra_args:
            raise ValueError(f"Extra arguments provided that are not in the default arguments: {extra_args}")

        # Merge the default arguments with the overrides
        merged_args = {**default_args, **override_args}

        # Initialize a new client only if the client_url has changed
        if self.client_url != client_url:
            self.client_url = client_url
            self.client = Client(client_url)

        # Call the API
        result = self.client.predict(**merged_args)
        return result
    
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

            # Convert the value to the appropriate Python type
            try:
                self.predict_args[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                self.predict_args[key] = value

    def __call__(self, code_str):
        """
        Allows the class instance to be called like a function.
        """
        self.extract_client_url(code_str)
        self.extract_predict_args_str(code_str)
        self.parse_predict_args()

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