import os
import shutil
import subprocess

from utils.logger import Logger
from utils.pathlib_utils import ensure_dir

def parse_output_dir(stdout_):
    lines = stdout_.splitlines()
    for line in lines:
        if line.startswith("[Logger] writing to PosixPath('"):
            # Extract the path inside the quotes
            path_start = line.find("'") + 1
            path_end = line.find("'", path_start)
            log_path = line[path_start:path_end]
            # The output dir is the parent of logs
            output_dir = os.path.dirname(os.path.dirname(log_path))
            return output_dir
    raise ValueError("Could not find logger output path in stdout")

def generate_sentiments():
    cmd = [
        "python3", "-m", "components.pipeline",
        "--ticker", "all-tickers",
        "--data-source", "fnspid",
        "--fine-tune",
        "--use-bodies"
    ]
    Logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        Logger.info(f"Error running command for {cmd} :\n{result.stderr}")
        return
    output_dir = parse_output_dir(result.stdout)
    timestamp = os.path.basename(output_dir)
    report_dir = f'report/fnspid-fine/{timestamp}/'
    os.makedirs(f'report/fnspid-fine', exist_ok=True)
    shutil.copytree(output_dir, report_dir, dirs_exist_ok=True)
    Logger.info(f"Copied output to {report_dir}")

if __name__ == "__main__":
    generate_sentiments()
