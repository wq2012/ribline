#!/bin/bash
set -o errexit

pip3 install --upgrade build twine
rm -rf dist

python3 -m build
twine upload dist/*
