import os 

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'output')

for path in [DATA_DIR, OUT_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

