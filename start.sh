#!/bin/bash
cp -r /nfs/my/solving-pde-with-nn /workspace/
pip install -r /workspace/solving-pde-with-nn/requirements.txt
jupyter lab --notebook-dir=/workspace/solving-pde-with-nn --NotebookApp.token='' --NotebookApp.allow_root=True --NotebookApp.ip=  > /var/jupyter.log 2>&1