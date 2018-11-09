import sys
import os

thirdparty_dir = os.path.dirname(os.path.realpath(__file__))

for d in os.listdir(thirdparty_dir):
    if d.startswith("_"):
        continue

    sys.path.append(os.path.join(thirdparty_dir, d))
