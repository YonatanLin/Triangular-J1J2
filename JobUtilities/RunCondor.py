import os
import sys
import subprocess
from pathlib import Path
from Main import CreateTriangularCaseDir

condor_script = sys.argv[1]
input_file = sys.argv[2]

def main():
    with open(input_file, "r") as file:
        input_file_lines = file.readlines()
     
    root_path = Path(os.getcwd())
    with open(condor_script, "r") as file:
        lines = file.readlines()
        new_root_line = "root_dir = " + str(root_path)
        lines = [new_root_line + "\n" if ("root_dir =" in line) else line for line in lines] 
        with open(condor_script, "w") as file:
            file.writelines(lines)

        #process = subprocess.Popen(["condor_submit", condor_script], stdout=subprocess.PIPE,
        #                           stderr=subprocess.PIPE, text=True)
        #stdout, stderr = process.communicate() 
        #print(stdout)
        #print(stderr)
main()
os.remove("jobs_input.txt")
