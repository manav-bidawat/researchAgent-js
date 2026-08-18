#!/bin/sh
# Stands in for the python executable: ignores the worker path it is given and runs the
# Node stub worker instead, so the CLI can be driven end to end without Python.
exec node "$(dirname "$0")/stub-worker.mjs"
