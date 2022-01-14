import os
from os import listdir
from os.path import isfile, join

path = 'instances/2'

files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    # os.system("python archetti_ext-heur.py instances/1/" + file + " > instances/1-archetti/" + file)
    os.system("python gdowska_exact.py instances/2/" + file + " 10000 > instances/2.1-gdowska/" + file)
    os.system("python gdowska_exact.py instances/2/" + file + " 10000 > instances/2.2-gdowska/" + file)
    os.system("python gdowska_exact.py instances/2/" + file + " 10000 > instances/2.3-gdowska/" + file)
    os.system("python gdowska_exact.py instances/2/" + file + " 10000 > instances/2.4-gdowska/" + file)
    os.system("python gdowska_exact.py instances/2/" + file + " 10000 > instances/2.5-gdowska/" + file)
