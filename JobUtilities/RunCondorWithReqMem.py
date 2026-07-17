import subprocess
import sys
from pathlib import Path
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--" + "rtype", type=str, required=True)
    parser.add_argument("--" + "cs_path", type=str, required=True)
    parser.add_argument("--" + "chi", type=int, required=True)
    parser.add_argument("--" + "area", type=int, required=True)
    parser.add_argument("--" + "finite", type=str, required=True)
    return parser


def update_and_submit_condor(script_path: str, new_memory: str):
    path = Path(script_path)

    if not path.is_file():
        print(f"Error: File '{script_path}' does not exist.")
        sys.exit(1)

    # Read the original lines
    try:
        lines = path.read_text().splitlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Modify or add the RequestMemory line
    memory_line_found = False
    updated_lines = []
    new_memory_line = f"RequestMemory = {new_memory}"

    for line in lines:
        # Match case-insensitive 'request_memory'
        match = "RequestMemory" in line
        if match and not memory_line_found:
            updated_lines.append(new_memory_line)
            memory_line_found = True
        elif not match:
            updated_lines.append(line)

    # If request_memory wasn't in the file, append it before the queue line
    if not memory_line_found:
        print("RequestMemory not found in script. Appending it.")
        # Try to insert before the 'queue' command if it exists
        queue_index = next(
            (
                i
                for i, l in enumerate(updated_lines)
                if l.strip().lower().startswith("queue")
            ),
            None,
        )
        if queue_index is not None:
            updated_lines.insert(queue_index, new_memory_line)
        else:
            updated_lines.append(new_memory_line)

    # Write the updates back to the file
    try:
        path.write_text("\n".join(updated_lines) + "\n")
        print(f"Successfully updated memory requirement to {new_memory}.")
    except Exception as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

    # Execute condor_submit
    print(f"Submitting {script_path} via condor_submit...")
    try:
        result = subprocess.run(
            ["condor_submit", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Submission successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during condor_submit:")
        print(e.stderr)
        sys.exit(e.returncode)


def dmrgMem(params_dict):
    chi, finite, area = params_dict["chi"], params_dict["finite"], params_dict["area"]
    chi_factor = 2e-7 * chi**2
    if finite=="finite":
        chi_factor *= 0.5
    mem_req = chi_factor * area * 1024
    return round(mem_req)

def gutzwillerMem(params_dict):
    chi, area = params_dict["chi"],  params_dict["area"]
    chi_factor = 15e-9 * chi**2
    return ((chi_factor * area) + 1) * 1024


if __name__ == "__main__":
    args = build_parser().parse_args()
    args_dict = vars(args)
    print(f"args dict: {args_dict}")

    sub_path = args_dict["cs_path"]
    run_type = args_dict["rtype"]
    
    req_mem = 1

    if run_type == "dmrg":
        req_mem = dmrgMem(args_dict)
    elif run_type == "gutzwiller":
        req_mem = gutzwillerMem(args_dict)
    else:
        print("Illegal run type")
        exit(1)

    update_and_submit_condor(sub_path, req_mem)
