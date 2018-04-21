#!/bin/bash

python ../../search_params/gen.py | awk '{print "nohup python run_mountaincarv0.py --alpha "$0" > tuning."$0" 2>&1 &"}' | sh

