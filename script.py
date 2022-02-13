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

    os.system("python gdowska_exact.py instances/1/" + file + " 1 > instances/Exaustivo/novo/" + file)

    """
    os.system("python gdowska_exact.py instances/1/" + file + " 1000 > instances/Gdowska-otimizado/novo/1000/1.1/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 1000 > instances/Gdowska-otimizado/novo/1000/1.2/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 1000 > instances/Gdowska-otimizado/novo/1000/1.3/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 1000 > instances/Gdowska-otimizado/novo/1000/1.4/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 1000 > instances/Gdowska-otimizado/novo/1000/1.5/" + file)

    os.system("python gdowska_exact.py instances/1/" + file + " 2500 > instances/Gdowska-otimizado/novo/2500/1.1/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 2500 > instances/Gdowska-otimizado/novo/2500/1.2/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 2500 > instances/Gdowska-otimizado/novo/2500/1.3/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 2500 > instances/Gdowska-otimizado/novo/2500/1.4/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 2500 > instances/Gdowska-otimizado/novo/2500/1.5/" + file)

    os.system("python gdowska_exact.py instances/1/" + file + " 5000 > instances/Gdowska-otimizado/novo/5000/1.1/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 5000 > instances/Gdowska-otimizado/novo/5000/1.2/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 5000 > instances/Gdowska-otimizado/novo/5000/1.3/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 5000 > instances/Gdowska-otimizado/novo/5000/1.4/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 5000 > instances/Gdowska-otimizado/novo/5000/1.5/" + file)

    os.system("python gdowska_exact.py instances/1/" + file + " 7500 > instances/Gdowska-otimizado/novo/7500/1.1/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 7500 > instances/Gdowska-otimizado/novo/7500/1.2/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 7500 > instances/Gdowska-otimizado/novo/7500/1.3/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 7500 > instances/Gdowska-otimizado/novo/7500/1.4/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 7500 > instances/Gdowska-otimizado/novo/7500/1.5/" + file)

    os.system("python gdowska_exact.py instances/1/" + file + " 10000 > instances/Gdowska-otimizado/novo/10000/1.1/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 10000 > instances/Gdowska-otimizado/novo/10000/1.2/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 10000 > instances/Gdowska-otimizado/novo/10000/1.3/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 10000 > instances/Gdowska-otimizado/novo/10000/1.4/" + file)
    os.system("python gdowska_exact.py instances/1/" + file + " 10000 > instances/Gdowska-otimizado/novo/10000/1.5/" + file)
    """

    i = i + 1
