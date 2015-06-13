#!/bin/sh
#!/bin/bash

# Stop on error
set -e
# Stop when undefined variable is ecountered
set -u
# Easier to debug errors
# set -o pipefail

file=${1:-}
if [[ -z "$file" ]]
then
    echo "Usage $0 FILE.cpp.template"
    exit 1
fi

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$( dirname $SCRIPT_DIR )

python3 $PROJECT_DIR/scripts/template_instantiator.py $file
