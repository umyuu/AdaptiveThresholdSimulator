from os.path import abspath
import sys

BASE_DIR = abspath('..')
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
