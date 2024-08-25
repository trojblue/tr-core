import unittest
from PIL import Image
from tr_core.utils.gradio_utils import GradioApiCaller, GradioCodeStrParser
import tempfile
import unibox as ub

class TestGradioUtils(unittest.TestCase):
    def setUp(self):
        self.code_str = """from gradio_client import Client, handle_file

client = Client("https://flag-celebration-manually-pan.trycloudflare.com/")
result = client.predict(
        image=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
        api_name="/demo_remove_exif"
)
print(result)"""
        self.file_url = "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
        self.gradio_caller = GradioApiCaller()
        self.gradio_parser = GradioCodeStrParser()

    def test_gradio_caller_with_file_url(self):
        url, args = self.gradio_parser(self.code_str)
        result = self.gradio_caller(url, args, {'image': f"FILE:{self.file_url}"})
        self.assertIsNotNone(result)  # Check that result is returned

    def test_gradio_caller_with_pil_image(self):
        pil_image = ub.loads(self.file_url)
        url, args = self.gradio_parser(self.code_str)
        result = self.gradio_caller(url, args, {'image': pil_image})
        self.assertIsNotNone(result)  # Check that result is returned

if __name__ == '__main__':
    unittest.main()