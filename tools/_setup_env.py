import os
import sys

my_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
root_dir = os.path.dirname(my_dir)
sys.path.insert(0, root_dir)
