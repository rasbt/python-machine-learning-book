import unittest
import os
import subprocess
import tempfile
import watermark
import nbformat


def run_ipynb(path):
    error_cells = []
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["python", "-m", "nbconvert", "--to",
                "notebook", "--execute", "--output",
                "--ExecutePreprocessor.kernel_name", "python"
                fout.name, path]
        subprocess.check_output(args)


class TestNotebooks(unittest.TestCase):

    def test_appendix_g_tensorflow_basics(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../appendix_g_tensorflow-basics.ipynb'))


if __name__ == '__main__':
    unittest.main()
