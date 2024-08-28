# tr-core
core modules for self codebase


## Utils


available functions:

```python
# str -> python types
from .string_utils import string_to_dtype       

# gradio code str -> python api calling 
from .gradio_utils import parse_gradio_api, call_gradio_api  

# comfyui api workflow -> gradio interface
from .comfyui_utils import workflow_to_iface
```

misc:

```python
from tr_core.utils import get_llava_response  # create payload and get response from llava model worker
```