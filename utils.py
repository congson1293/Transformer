import os

def mkdir(dir):
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except: pass
