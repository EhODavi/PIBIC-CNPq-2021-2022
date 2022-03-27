from os import listdir
from os.path import isfile, join

path = 'instances/a/montecarlo/1'

files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    arquivo1 = open('instances/a/montecarlo/1/' + file, 'r')
    conteudo_arquivo1 = []

    arquivo2 = open('instances/a/montecarlo/2/' + file, 'r')
    conteudo_arquivo2 = []

    arquivo3 = open('instances/a/montecarlo/3/' + file, 'r')
    conteudo_arquivo3 = []

    arquivo4 = open('instances/a/montecarlo/4/' + file, 'r')
    conteudo_arquivo4 = []

    arquivo5 = open('instances/a/montecarlo/5/' + file, 'r')
    conteudo_arquivo5 = []

    for linha in arquivo1:
        linha = linha.strip()
        conteudo_arquivo1.append(linha)

    for linha in arquivo2:
        linha = linha.strip()
        conteudo_arquivo2.append(linha)

    for linha in arquivo3:
        linha = linha.strip()
        conteudo_arquivo3.append(linha)

    for linha in arquivo4:
        linha = linha.strip()
        conteudo_arquivo4.append(linha)

    for linha in arquivo5:
        linha = linha.strip()
        conteudo_arquivo5.append(linha)

    linha1 = conteudo_arquivo1[3].split()
    linha2 = conteudo_arquivo2[3].split()
    linha3 = conteudo_arquivo3[3].split()
    linha4 = conteudo_arquivo4[3].split()
    linha5 = conteudo_arquivo5[3].split()

    numero1 = float(linha1[0])
    numero2 = float(linha2[0])
    numero3 = float(linha3[0])
    numero4 = float(linha4[0])
    numero5 = float(linha5[0])

    print(round((numero1 + numero2 + numero3 + numero4 + numero5) / 5, 2))

    arquivo1.close()
    arquivo2.close()
    arquivo3.close()
    arquivo4.close()
    arquivo5.close()
