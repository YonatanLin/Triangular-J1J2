#!/bin/bash
# Send a job computing <psi1|psi2> of the wavefunctions in two results directories (OverlapRunner.py).
#   ./SubmitOverlap.sh <dir1> <dir2> [mem=<MB>] [cpus=<n>] [psi1=<pkl>] [psi2=<pkl>]
# The directories may be relative or have a trailing slash. Result: <dir1>/overlap_with_<dir2 name>.txt
set -e
if [ $# -lt 2 ]; then
    echo "usage: $0 <dir1> <dir2> [mem=<MB>] [cpus=<n>] [psi1=<pkl>] [psi2=<pkl>]"
    exit 1
fi
dir1=$(realpath "$1")
dir2=$(realpath "$2")
shift 2
for d in "$dir1" "$dir2"; do
    [ -d "$d" ] || { echo "no such directory: $d"; exit 1; }
done
condor_submit "$(dirname "$(realpath "$0")")/SendJobsOverlap.sub" dir1="$dir1" dir2="$dir2" "$@"
