#!/bin/sh


pip  uninstall -y cbnn
hatch test
hatch build
pip install dist/cbnn-0.0.1-py3-none-any.whl 

