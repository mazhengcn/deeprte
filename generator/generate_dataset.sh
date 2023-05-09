#!/bin/bash

# module add matlab/2020a

cd /workspaces/deeprte/generator/2d-sweeping

matlab -nodisplay -r "run generator.m; exit"
