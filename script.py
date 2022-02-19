import os
from os import listdir
from os.path import isfile, join

path = 'instances/1'

files = [f for f in listdir(path) if isfile(join(path, f))]

i = 1
total = len(files)

for file in files:
    print(f"Estou no arquivo {i} de {total}...")

    # os.system("python archetti_ext-heur.py instances/1/" + file + " > instances/Archetti/novo/" + file)

    os.system("python gdowska_exact.py instances/1/" + file + " > instances/Exaustivo/novo/" + file)

    """
    os.system("python gdowska_exact.py instances/1/" + file + " > instances/Gdowska/novo/1.1/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " > instances/Gdowska/novo/1.2/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " > instances/Gdowska/novo/1.3/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " > instances/Gdowska/novo/1.4/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " > instances/Gdowska/novo/1.5/" + file)
    """

    i = i + 1
