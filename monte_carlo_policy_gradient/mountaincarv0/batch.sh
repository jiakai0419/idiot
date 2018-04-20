#!/bin/bash

python ../../search_params/gen.py | awk '{print "python run_mountaincarv0.py --alpha "$0" > tunning."$0" &"}' | sh

