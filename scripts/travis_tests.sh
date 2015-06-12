#!/bin/sh
#!/bin/bash

# Stop on error
set -e
# Stop when undefined variable is ecountered
set -u
# Easier to debug errors
set -o pipefail

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

pushd $SCRIPT_DIR/../build
    cmake -DWITH_CUDA=FALSE ..
    make -j9 run_tests
    make -j9
    result=$?
    rm -r ./*
    echo "Unit tests completed : $result"
popd
exit $result
