from typing import Dict, Any, Iterator, Tuple, List, Tuple, Callable

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import inspect

import gradio as gr


class Workflow:
    def __init__(self, raw_json: Dict[str, Any]):
        self.raw_json = raw_json
        self._modifiable_keys = self._extract_modifiable_keys()

    def _extract_modifiable_keys(self) -> Dict[str, Tuple[str, Any]]:
        """Extracts and returns a dictionary of modifiable keys and their content."""
        modifiable_keys = {}
        for node_id, node_data in self.raw_json.items():
            if node_data['class_type'] == 'TaggedAny':
                tag = node_data['inputs'].get('tag')
                content = node_data['inputs'].get('content')
                if tag and content:
                    modifiable_keys[tag] = (node_id, content)
        return modifiable_keys

    def get_modifiable_keys(self) -> Dict[str, Any]:
        """Returns a dictionary of modifiable keys and their content."""
        return {key: value[1] for key, value in self._modifiable_keys.items()}

    def update_modifiable_keys(self, overrides: Dict[str, Any]) -> None:
        """Updates the modifiable keys using the given overrides."""
        for key, value in overrides.items():
            if key in self._modifiable_keys:
                node_id, _ = self._modifiable_keys[key]
                self.raw_json[node_id]['inputs']['content'] = value
                self._modifiable_keys[key] = (node_id, value)

    def export(self) -> Dict[str, Any]:
        """Exports the workflow as a dictionary."""
        return self.raw_json

    def __getitem__(self, key: str) -> Any:
        """Allows direct access to raw_json like a dictionary."""
        return self.raw_json[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allows direct setting of items in raw_json like a dictionary."""
        self.raw_json[key] = value
        self._modifiable_keys = self._extract_modifiable_keys()  # Refresh modifiable keys

    def __delitem__(self, key: str) -> None:
        """Allows direct deletion of items in raw_json like a dictionary."""
        del self.raw_json[key]
        self._modifiable_keys = self._extract_modifiable_keys()  # Refresh modifiable keys

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Allows iteration over the raw_json as key-value pairs."""
        return iter(self.raw_json.items())

    def __len__(self) -> int:
        """Returns the length of the raw_json."""
        return len(self.raw_json)

    def __repr__(self) -> str:
        """Custom representation for the Workflow class."""
        return f"<Workflow with {len(self.raw_json)} nodes> | modifiable keys: {list(self._modifiable_keys.keys())}"


class WorkflowExecutor:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data, method="POST")
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_images(self, ws, prompt):
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  # Execution is done
            else:
                continue  # Previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for o in history['outputs']:
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                if 'images' in node_output:
                    images_output = []
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        pil_image = Image.open(io.BytesIO(image_data))
                        images_output.append(pil_image)
                    output_images[node_id] = images_output

        return output_images

    def run_workflow(self, workflow_json:dict):
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        images = self.get_images(ws, workflow_json)
        return images


class WorkflowGradioGenerator:

    DEFAULT_COMMON_ORDERS = [
        'api_positive', 'api_negative', 'api_width', 'api_height', 'api_batchsize', 'api_steps',
    ]

    def __init__(self, common_orders: List[str] = None):
        """
        Initializes the WorkflowGradioGenerator with an optional list of common keys.
        """
        self.common_orders = common_orders if common_orders else self.DEFAULT_COMMON_ORDERS

    def _extract_gradio_inputs(self, modifiable_keys: Dict[str, Any], hidden_params: List[str] = None) -> Tuple[List[gr.components.Component], Dict[str, Any]]:
        """
        Dynamically generate Gradio input components based on modifiable keys.
        Rearranges inputs by having the common_keys on top, sorted by their order.
        """
        hidden_params = hidden_params or []
        gr_inputs = []
        input_mapping = {}

        # Sort keys according to common_orders, with non-common keys following
        sorted_keys = sorted(
            [k for k in modifiable_keys.keys() if k not in hidden_params],
            key=lambda k: (k not in self.common_orders, self.common_orders.index(k) if k in self.common_orders else float('inf'))
        )
        
        for key in sorted_keys:
            value = modifiable_keys[key]
            if isinstance(value, int):
                gr_input = gr.Number(label=key, value=value)
            elif isinstance(value, float):
                gr_input = gr.Number(label=key, value=value)
            elif isinstance(value, str):
                gr_input = gr.Textbox(label=key, value=value)
            else:
                gr_input = gr.Textbox(label=key, value=str(value))  # fallback for other types

            gr_inputs.append(gr_input)
            input_mapping[key] = gr_input

        return gr_inputs, input_mapping

    def _run_workflow_with_overrides(self, workflow: Any, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the workflow with the given overrides and return the result.
        """
        workflow.update_modifiable_keys(overrides)
        api_wf = dict(workflow)        

        executor = WorkflowExecutor()
        return executor.run_workflow(api_wf)

    def _generate_gradio_function(self, workflow: Any, input_keys: List[str]) -> Callable:
        """
        Generates a Gradio function that accepts fixed arguments based on input_keys.
        """
        def gradio_fn(*args):
            """
            Function to be executed by Gradio with fixed arguments.
            """
            overrides = dict(zip(input_keys, args))
            res = self._run_workflow_with_overrides(workflow, overrides)

            images = []
            for key, img_list in res.items():
                if isinstance(img_list, list):
                    images.extend(img_list)

            return images, res

        # Set the function signature to match the input keys
        sig = inspect.signature(gradio_fn)
        new_params = [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD) for k in input_keys]
        gradio_fn.__signature__ = sig.replace(parameters=new_params)

        return gradio_fn
    
    
    def __call__(self, 
                 workflow: Any, 
                 hidden_params: List[str] = None, 
                 title: str = "Dynamic Workflow Runner", 
                 description: str = "Modify and run the workflow dynamically using Gradio."
                 ) -> gr.Interface:
        """
        Generates a Gradio interface for the given workflow, optionally hiding certain parameters.
        """
        modifiable_keys = workflow.get_modifiable_keys()
        gr_inputs, input_mapping = self._extract_gradio_inputs(modifiable_keys, hidden_params=hidden_params)

        # Generate the actual function to be used in the Gradio interface
        input_keys = list(input_mapping.keys())
        gradio_fn = self._generate_gradio_function(workflow, input_keys)

        iface = gr.Interface(
            fn=gradio_fn,
            inputs=gr_inputs,
            outputs=[gr.Gallery(label="Generated Images"), gr.JSON(label="Raw Response")],
            title=title,
            description=description,
        )

        return iface


def workflow_to_iface(raw_workflow:dict, hidden_params:list[str]=[]) -> gr.Interface:
    """
    Converts a Comfy-generated workflow to a Gradio interface.

    Parameters:
        raw_workflow (dict): The workflow to convert, using TaggedAny nodes as inputs
        hidden_params (list[str]): The parameters to hide in the interface.
    """
    workflow = Workflow(raw_workflow)
    generator = WorkflowGradioGenerator()
    iface = generator(workflow, hidden_params=hidden_params)
    return iface

    
def demo():
    
    import unibox as ub

    wf = ub.loads("/rmt/yada/dev/tr-core/notebooks/vpred_api_default_workflow_api.json")

    workflow = Workflow(wf)
    modifiable_keys = workflow.get_modifiable_keys()

    # Update the modifiable keys
    overrides = {'api_batchsize': '2',
    'api_positive': 'new positive content',
    'api_negative': 'new negative content',
    'api_width': '896',
    'api_height': '1152',
    'api_cfg': '0.9',
    'api_sampler_name': 'euler_cfg_pp',
    'api_steps': 28
    }


    workflow.update_modifiable_keys(overrides)
    api_wf = dict(workflow)  # convert to api-compatible dictionary

    from training_flow.utils.run_comfy_api import WorkflowExecutor

    executor = WorkflowExecutor()
    res = executor.run_workflow(api_wf)




if __name__ == "__main__":
    demo()