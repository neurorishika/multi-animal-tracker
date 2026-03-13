#!/usr/bin/env python
import os
import sys

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, src_dir)

from multi_tracker.tools.data_sieve.gui import main

if __name__ == "__main__":
    main()
