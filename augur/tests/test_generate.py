from pathlib import Path
from augur.generate import generate


def test_generate():
    base_path = Path(__file__).parent
    generate(f'{base_path}/test.yaml')
