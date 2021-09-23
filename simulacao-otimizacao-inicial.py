import math
import random


def sigmoid(B0, B1, C):
    return 1 / (1 + math.exp(-(B0 + B1 * C)))


N = 100  # Número de entregas
F = 10  # Custo de entrega para a empresa
M = 10  # Número de Entregadores Ocasionais
B0 = -5  # Constante B0 da função sigmoid
B1 = 1  # Constante B1 da função sigmoid

menor_custo_total = math.inf  # Menor custo total para a empresa
menor_C = None  # Menor valor a ser oferecido para os entregadores ocasionais

for i in range(100000):
    random.seed(random.randint(0, 100000))
    C = random.uniform(0.0, F)  # Custo de entrega que a empresa pode oferecer para os entregadores ocasionais

    P = sigmoid(B0, B1, C)  # Probabilidade associada ao valor C escolhido aleatoriamente anteriormente

    K = 0  # Quantidade de entregadores ocasionais que aceitam fazer uma entrega

    for j in range(M):
        random.seed(random.randint(0, 100000))
        p = random.uniform(0.0, 1.0)  # Gerando um número aleatório para cada entregador ocasional

        # O valor gerado (p) para o entregador ocasional for menor ou igual a P, isso quer dizer que ele vai aceitar
        # fazer a entrega
        if p <= P:
            K += 1

    custo_total = F * (N - K) + C * K

    if custo_total < menor_custo_total:
        menor_custo_total = custo_total
        menor_C = C

print(f"C = {menor_C}")
