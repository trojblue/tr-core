# tr-core
core modules for self codebase


## Utils


available functions:

```python
from .string_utils import string_to_dtype       # str -> python types
from .gradio_utils import parse_gradio_api, call_gradio_api  # gradio code str -> python api calling 
from .comfyui_utils import workflow_to_iface    # comfyui api workflow -> gradio interface
```

misc:

```python
from tr_core.utils import get_llava_response  # create payload and get response from llava model worker
```