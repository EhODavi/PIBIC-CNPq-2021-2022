import math, random


def sigmoid(B0, B1, C):
    return 1 / (1 + math.exp(-(B0 + B1 * C)))


N = 100  # Número de entregas
F = 10  # Custo de entrega para a empresa
M = 10  # Entregadores Ocasionais
B0 = -5
B1 = 1
menor_custo = math.inf
menor_C = -1


for i in range(100000):
    random.seed(random.random())
    C = random.uniform(0.0, F)  # Custo de entrega que a empresa pode oferecer para os entregadores ocasionais
    P = sigmoid(B0, B1, C)

    K = 0

    for j in range(M):
        random.seed(random.random())
        p = random.random()

        if p >= P:
            K += 1

    custo_total = F * (N - K) + C * K

    if custo_total < menor_custo:
        menor_custo = custo_total
        menor_C = C

print(f"C = {menor_C}")
