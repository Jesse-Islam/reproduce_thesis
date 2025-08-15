#!/bin/sh


mv examples/ ../
pip  uninstall -y gosip
#hatch test
hatch build
mv ../examples/ .
pip install dist/gosip-0.0.1-py3-none-any.whl

