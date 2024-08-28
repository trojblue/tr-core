# tr-core
core modules for self codebase


## Utils


available functions:

```python
from tr_core.utils import StringConverter  # str -> python types
from tr_core.utils import workflow_to_iface # comfyui api workflow -> gradio interface
from tr_core.utils import GradioCodeStrParser, GradioApiCaller  # gradio code str -> python api calling 
```

misc:

```python
from tr_core.utils import get_llava_response  # create payload and get response from llava model worker
```