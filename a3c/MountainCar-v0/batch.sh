#!/bin/bash

python ../../search_params/gen.py | awk '{print "nohup python main.py --lr "$0" --num-processes 1 > tuning."$0" 2>&1 &"}' | sh

