# stolen from campaigns runner

import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor


def run_command(param_file, main_cli_path):
    """
    Run the main_cli.py script with the specified parameter file.

    Args:
        param_file (str): Path to the parameter file.
        main_cli_path (str): Path to the main_cli.py script.
    """
    command = [sys.executable, main_cli_path, "-p", param_file]
    subprocess.run(command)


def main():
    """
    Run a campaign by executing main_cli.py for each parameter file in the campaign's input directory using multiple threads.

    Args:
        campaign_name (str): Name of the campaign to run.
    """
    # Path to the working directory
    workfolder = os.path.dirname(os.path.abspath(__file__))
    main_cli_path = os.path.join(workfolder, "../sharc/ngso_to_gso.py")

    # Campaign directory
    campaign_folder = os.path.join(
        workfolder, "input",
    )

    # List of parameter files
    parameter_files = [
        os.path.join(campaign_folder, f) for f in os.listdir(
            campaign_folder,
        ) if f.endswith('.yaml')
    ]

    if len(parameter_files) == 0:
        raise ValueError(
            f"No parameter files were found in {campaign_folder}"
        )

    # Number of threads (adjust as needed)
    num_threads = min(len(parameter_files), os.cpu_count())

    # Run the commands in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(
            run_command, parameter_files, [
                main_cli_path,
            ] * len(parameter_files),
        )


if __name__ == "__main__":
    main()
