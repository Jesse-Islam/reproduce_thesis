#!/bin/sh


pip  uninstall -y gosip
hatch test
hatch build
pip install dist/gosip-0.9-py3-none-any.whl

