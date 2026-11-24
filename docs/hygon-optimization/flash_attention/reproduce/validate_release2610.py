"""Run the native-HIP repository regressions and full workbook stability suite."""

import argparse
import importlib
import os
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--environment-root", type=Path, required=True)
args = parser.parse_args()
root = args.environment_root.resolve()
repo = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / "fg-compiler36"))
sys.path.insert(0, str(repo / "src"))
os.environ["GEMS_VENDOR"] = "hygon"
os.environ["AMDGCN_USE_BUFFER_OPS"] = "0"
os.environ["ROCM_PATH"] = "/public/software/compiler/dtk-26.04"
os.environ["HIP_PATH"] = "/public/software/compiler/dtk-26.04/hip"
torch = importlib.import_module("torch")
pytest = importlib.import_module("pytest")
torch.set_num_threads(16)
torch.set_grad_enabled(False)
assert torch.cuda.device_count() == 1
assert os.environ["HIP_VISIBLE_DEVICES"] in ("3", "4")
assert os.environ.get("APPTAINER_CONTAINER")
assert (
    pytest.main(
        ["-q", str(repo / "tests/test_flash_attention_hygon_state.py"), "--ref", "cpu"]
    )
    == 0
)
runpy.run_path(
    str(Path(__file__).with_name("validate_workbook_stability_cuda.py")),
    run_name="__main__",
)
