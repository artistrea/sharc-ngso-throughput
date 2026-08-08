from sharc.parameters.parameters import Parameters
from pathlib import Path

if __name__ == "__main__":
    param_file = Path("./test_params.yaml").resolve()

    parameters = Parameters()
    parameters.set_file_name(param_file)
    parameters.read_params()

