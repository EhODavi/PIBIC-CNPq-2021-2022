import os
from os import listdir
from os.path import isfile, join

path = 'instances/a1'

files = [f for f in listdir(path) if isfile(join(path, f))]

i = 1
total = len(files)

for file in files:
    print(f"Estou no arquivo {i} de {total}...")

    # os.system("python archetti_ext-heur.py instances/1/" + file + " > instances/Archetti/novo/" + file)

    # os.system("python gdowska_exact.py instances/1/" + file + " > instances/Exaustivo/novo/" + file)

    os.system("python gdowska_exact.py instances/a1/" + file + " > instances/a/montecarlo/1/" + file)
    os.system("python gdowska_exact.py instances/a1/" + file + " > instances/a/montecarlo/2/" + file)
    os.system("python gdowska_exact.py instances/a1/" + file + " > instances/a/montecarlo/3/" + file)
    os.system("python gdowska_exact.py instances/a1/" + file + " > instances/a/montecarlo/4/" + file)
    os.system("python gdowska_exact.py instances/a1/" + file + " > instances/a/montecarlo/5/" + file)

    i = i + 1
