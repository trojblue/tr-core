from typing import Dict, Any, Iterator, Tuple

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