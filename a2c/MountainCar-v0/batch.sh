#!/bin/bash

python ../../search_params/gen.py | awk '{print "nohup python main.py --lr "$0" > /dev/null 2>&1 &"}' | sh

