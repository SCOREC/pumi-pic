#!/bin/bash
module load matlab
matlab -nosplash -noFigureWindows -r "try; run('generate_all_graphs.m'); catch; end; quit"
