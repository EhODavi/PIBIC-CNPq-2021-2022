import random
from gurobipy import *
from collections import namedtuple, defaultdict
from random import choice
from monte_carlo_tree_search import MCTS, Node
from read_santini import read_santini

_Gdowska = namedtuple("Gdowska", "free pf oc winner terminal")

tsp_obj = None
C_global = None
c_global = None
q_global = None
p_global = None
r_global = None
Q_global = None


class LastMile(_Gdowska, Node):
    def find_children(self):
        if self.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in oc or in pf
        return {self.make_move(-1, True), self.make_move(-1, False)}

    def find_random_child(self):
        if self.terminal:
            return None  # If the game is finished then no moves can be made
        x = choice([True,False])
        return self.make_move(-1, x)

    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        if self.winner is None:
            return 0.5  # Board is a tie
        elif self.winner == False:
            return 0
        elif self.winner == True:
            return 1
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    def is_terminal(self):
        return self.terminal

    def make_move(self, index, is_oc):
        free = list(self.free)
        cust = free.pop(index)
        free = tuple(free)
        if is_oc:
            oc = frozenset(self.oc | {cust})
            pf = self.pf
        else:
            oc = self.oc
            pf = frozenset(self.pf | {cust})
        is_terminal = (len(free) == 0)
        if is_terminal:
            winner = _find_winner(free, pf, oc)
        else:
            winner = -111   # should raise an error if used
        return LastMile(free=free, pf=pf, oc=oc,
                        winner=winner, terminal=is_terminal)

    def to_pretty_string(self):
        return f"free={self.free}, pf={self.pf}, oc={self.oc}"


def archetti(C, c, q, r, Q, fix):
    """modified archetti's model to consider that demand can
    be outsourced to occasional couriers
    Parameters:
        - C: list of customers [depot is 0]
        - c[i,j]: cost of edge (i,j)
        - q[i]: demand for customer i
        - r[i]: cost for outsourcing delivery to i
        - Q: vehicle's capacity
        - fix[i]: 1 if customer i is outsourced, 0 otherwise
    Returns constructed gurobi model.
    """

    # VARIABLES
    model = Model("modified archetti")
    w = {}  # w[i] = 1 if customer i is visited by occasional driver
    x = {}  # x[i,j] = 1 if regular vehicle traverses ij
    y = {}  # y[i,j] load a regular vehicle carries when traversing ij
    z = {}  # z[i] = 1 if customer i is visited by regular vehicle
    for i in [0]+C:
        for j in [0]+C:
            if j == i:
                continue
            x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))
            y[i,j] = model.addVar(vtype="C", lb=0, name="y(%s,%s)"%(i,j))
    for i in C:
        z[i] = model.addVar(vtype="B", name="z(%s)"%(i))
        w[i] = model.addVar(vtype="B", name="w(%s)"%(i))

    # CONSTRAINTS
    # flow conservation and demand satisfaction
    model.addConstr(quicksum(x[0,j] for j in C) - quicksum(x[j,0] for j in C) == 0, "Depot")
    model.addConstr(quicksum(y[j,0] for j in C) - quicksum(y[0,j] for j in C) ==
                    quicksum(-q[i]*z[i] for i in C), "FlowDepot")
    for i in C:
        model.addConstr(quicksum(x[i,j] for j in [0]+C if j != i) == z[i], "Out(%d)"%i)
        model.addConstr(quicksum(x[j,i] for j in [0]+C if j != i) == z[i], "In(%d)"%i)
        model.addConstr(quicksum(y[j,i] for j in [0]+C if j != i) -
                        quicksum(y[i,j] for j in [0]+C if j != i) == q[i]*z[i], "Flow(%d)"%i)

    # vehicle capacity
    for i in [0]+C:
        for j in [0]+C:
            if j == i:
                continue
            model.addConstr(y[i,j] <= Q*x[i,j], "Cap(%d,%d)"%(i,j))
    for i in C:
        model.addConstr(y[i,0] == 0, "EmptyReturn(%d)"%i)

    # customer service
    for i in C:
        model.addConstr(w[i] == fix[i], "Fixed(%d)"%(i))

    for i in C:
        model.addConstr(w[i] + z[i] == 1, "Service(%d)"%i)

    # objective
    model.setObjective(quicksum(c[i,j]*x[i,j] for i in [0]+C for j in [0]+C if j != i) +
                       quicksum(r[i]*w[i] for i in C),
                       GRB.MINIMIZE)

    model.__data = w,x,y,z
    model.Params.OutputFlag = 0 # silent mode
    return model


# globals
MIPCACHE = {}


def cache_archetti(C, c, q, r, Q, A):
    """
    check if current archetti model has already been solved; if not, solve and cache it
    Parameters: see archetti
        - A: subset of customers to be outsourced
    Returns the optimum objective value.
    """
    key = frozenset(A)
    if key in MIPCACHE:
        return MIPCACHE[key]
    fix = {i:(1 if i in A else 0) for i in C}
    model = archetti(C, c, q, r, Q, fix)
    model.optimize()
    w,x,y,z = model.__data
    # print("Optimal solution:", model.objval)
    # print("w=", [(i,j) for (i,j) in w if w[i,j].X > .5])
    # print("x=", [(i,j) for (i,j) in x if x[i,j].X > .5])
    # print("z=", [i for i in z if z[i].X > .5])
    # print("y=")
    # for (i,j) in y:
    #     if y[i,j].X > 0:
    #         print("y[%s,%s] = %g" % (i,j,y[i,j].X))
    # for i,c in enumerate(cycles([0]+C,[(i,j) for (i,j) in x if x[i,j].X > .5])):
    #     print("\t{}\t{}".format(i,c))
    z = model.objval
    MIPCACHE[key] = z
    return z


# globals
PATCACHE = {}


def eval_archetti(C, c, q, p, r, Q, A):
    """
    evaluate archetti model with probability of couriers not accepting a delivery
    Parameters: see cachet_archetti
        - p[i,k]: probability of k accepting a delivery to i
    Returns the optimum objective value and the list of edges used.
    """
    key = frozenset(A)
    if key in PATCACHE:
        return PATCACHE[key]
    a = list(A)
    expect = 0
    for seq in itertools.product([0,1], repeat=len(a)):  # seq is 00...0, 00...1, ..., 11...1
        prod = 1
        pat = set()
        for j in range(len(seq)):
            i = a[j]
            if seq[j] == 1:  # DELIVERY ACCEPTED
                pat.add(i)
                prod *= p[i]
            else:  # DELIVERY NOT ACCEPTED
                prod *= (1-p[i])
        z = cache_archetti(C, c, q, r, Q, pat)
        s = "{}" if len(pat)==0 else str(pat)
        # print("P{}={}; z={}".format(s,prod,z))
        expect += prod * z
    # print("E{} = {}".format(A,expect))
    PATCACHE[key] = expect
    return expect

"""
def make_data_random(n):
    make_data_random: prepare all relevant data randomly:
    compute matrix distance based on euclidean distance
    between n customers, m occasional drivers.
    Returns tuple (C, c, q, K, p, alpha, probmax, v0, Q), where:
        - C: list of customers [depot is 0]
        - c[i,j]: cost of edge (i,j)
        - q[i]: demand for customer i
        - p[i]: probability of an outsource to i being accepted
        - r[i]: cost for outsourcing delivery to i
        - Q: vehicle's capacity
    
    #distance = lambda x1, y1, x2, y2: math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance = lambda x1, y1, x2, y2: int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    # CUSTOMERS
    C = list(range(1,n+1))
    #x = {i: random.random() for i in [0] + C}
    #y = {i: random.random() for i in [0] + C}
    x = {i:random.randint(0, 100) for i in [0]+C}
    y = {i:random.randint(0, 100) for i in [0]+C}
    c, q = {}, {}  # c[i,j] -> cost matrix, q[i] -> demand
    p, r = {}, {}  # p[i], r[i] -> prob/cost for outsourcing to i
    beta = {}      # beta[i,k] = 1 if i can be served by k
    for i in [0]+C:
        #q[i] = random.randint(10,20)  # demand for customer i
        q[i] = 1
        for j in [0]+C:
            c[i,j] = distance(x[i],y[i],x[j],y[j])
            c[j,i] = distance(x[i],y[i],x[j],y[j])

    # COURIERS
    for i in [0] + C:
        p[i] = random.random()  # probability of an outsource to i being accepted
        r[i] = random.randint(5, 10) # cost for outsourcing delivery to i

    # VEHICLES
    #Q = 50  # vehicle's capacity
    Q = len(C)  # vehicle's capacity

    return C, c, q, p, r, Q, x, y


def make_data_read(n):
    make_data_read: read all relevant data:
    compute matrix distance based on euclidean distance
    between n customers, m occasional drivers.
    Returns tuple (C, c, q, K, p, alpha, probmax, v0, Q), where:
        - C: list of customers [depot is 0]
        - c[i,j]: cost of edge (i,j)
        - q[i]: demand for customer i
        - p[i]: probability of an outsource to i being accepted
        - r[i]: cost for outsourcing delivery to i
        - Q: vehicle's capacity
    
    distance = lambda x1, y1, x2, y2: math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # CUSTOMERS
    C = list(range(1,n+1))

    x = dict()
    y = dict()

    for i in [0]+C:
        xi = float(input(f"x{i} = "))
        yi = float(input(f"y{i} = "))
        x.update({i:xi})
        y.update({i:yi})

    c, q = {}, {}  # c[i,j] -> cost matrix, q[i] -> demand
    p, r = {}, {}  # p[i], r[i] -> prob/cost for outsourcing to i
    beta = {}      # beta[i,k] = 1 if i can be served by k
    for i in [0]+C:
        q[i] = int(input(f"Demanda do cliente {i} = "))
        for j in [0]+C:
            c[i,j] = distance(x[i],y[i],x[j],y[j])
            c[j,i] = distance(x[i],y[i],x[j],y[j])

    # COURIERS
    for i in [0] + C:
        p[i] = float(input(f"Probabilidade de uma entrega para o cliente {i} ser aceita = "))
        r[i] = float(input(f"Custo de se entregar para o cliente {i} = "))

    # VEHICLES
    Q = 50  # vehicle's capacity

    return C, c, q, p, r, Q, x, y
"""


def main():
    import time
    import itertools
    import sys

    # inicio = time.time()

    global tsp_obj, C_global, c_global, q_global, p_global, r_global, Q_global

    if len(sys.argv) != 3:
        print("usage: {} filename N".format(sys.argv[0]))
        print("    where")
        print("      - filename is the instance file (Santini's format)")
        print("      - N is the number of iterations")
        exit(-1)

    C, c, x, y, p, r = read_santini(sys.argv[1])

    C.pop()

    for i in range(len(C)):
        C[i] = int(C[i])

    n = len(C)
    Q = n

    new_x = {}
    new_y = {}

    for key, value in x.items():
        new_key = int(key) - 1
        new_x[new_key] = value

    for key, value in y.items():
        new_key = int(key) - 1
        new_y[new_key] = value

    x = new_x
    y = new_y

    q = {}

    for i in range(n + 1):
        q[i] = 1

    new_r = {}

    for key, value in r.items():
        new_key = int(key) - 1
        new_r[new_key] = value

    r = new_r

    new_p = {}

    for key, value in p.items():
        new_key = int(key) - 1
        new_p[new_key] = value

    p = new_p

    c = {}

    distance = lambda x1, y1, x2, y2: int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    for i in [0] + C:
        for j in [0] + C:
            c[i, j] = distance(x[i], y[i], x[j], y[j])
            c[j, i] = distance(x[i], y[i], x[j], y[j])

    """
    opcao = input('Deseja ler uma instância (1/0)?\n')

    if opcao == "1":
        n = int(input("Número de Clientes = "))
        C, c, q, p, r, Q, x, y = make_data_read(n)
    else:
        # initialize the random generator
        try:
            r1 = int(sys.argv[1])
        except:
            r1 = int(time.time())
        print("Random seeds: r1=%d" % (r1))
        random.seed(r1)

        try:
            n = int(sys.argv[2])
        except:
            n = 50  # customers to serve

        C, c, q, p, r, Q, x, y = make_data_random(n)
    """

    C_new = []

    for ci in C:
        Ai = []

        Ai.append(ci)

        result = cache_archetti(C, c, q, r, Q, Ai)

        C_new.append((result, ci))

    C_new.sort()

    C = []

    for i in C_new:
        C.append(i[1])

    """
    print(f"n={n}")
    print(f"C={C}")
    print(f"x={x}")
    print(f"y={y}")
    print(f"q={q}")
    print(f"p={p}")
    print(f"r={r}")
    print(f"Q={Q}")
    """

    C_global = C
    c_global = c
    q_global = q
    p_global = p
    r_global = r
    Q_global = Q

    # solution with no outsourcing
    tsp_obj = cache_archetti(C, c, q, r, Q, [])
    # zmin = tsp_obj
    # amin = []
    # print("SOLUÇÃO SEM ENTREGADORES OCASIONAIS -> {:.4g}".format(tsp_obj))

    tree = MCTS()
    board = new_lastmile(C)

    for _ in range(int(sys.argv[2])):
        tree.do_rollout(board)

    global OC_CACHE
    # global PATCACHE

    maior_n = -math.inf
    conjunto = []

    for key, value in OC_CACHE.items():
        print(f"{key} = {len(value)}")

        if len(value) > maior_n:
            conjunto = key
            maior_n = len(value)

    amontecarlo = []
    zmontecarlo = tsp_obj

    for key, value in OC_CACHE.items():
        valor_medio = sum(value) / len(value)

        if valor_medio < zmontecarlo:
            amontecarlo = []

            for val in key:
                amontecarlo.append(val)

            zmontecarlo = valor_medio

    """
    for key, value in PATCACHE.items():
        if value < zmontecarlo:
            amontecarlo = []

            for val in key:
                amontecarlo.append(val)

            zmontecarlo = value
    """

    amontecarlo.sort()
    zmontecarlo = eval_archetti(C, c, q, p, r, Q, amontecarlo)

    # print("MIN - MONTE CARLO - OTIMIZADO")
    # print("{:.4g} <- {}".format(zmontecarlo, amontecarlo))

    """
    C = []
    C.extend(range(1, n + 1))
    w = len(C)
    masks = [1 << i for i in range(w)]
    for i in range(1 << w):
        A = [ss for mask, ss in zip(masks, C) if i & mask]
        z = eval_archetti(C, c, q, p, r, Q, A)
        # print("{:.4g} <- {}".format(z, A))
        if z < zmin:
            zmin = z
            amin = A
    print("MIN - EXAUSTIVO")
    print("{:.4g} <- {}".format(zmin, amin))
    sys.stdout.flush()

    conjunto = []
    fix = {i: (1 if i in conjunto else 0) for i in C}
    modelAll = archetti(C, c, q, r, Q, fix)

    fix = {i: (1 if i in amin else 0) for i in C}
    modelA = archetti(C, c, q, r, Q, fix)

    return n, x, y, amin, modelAll, modelA

    fim = time.time()
    print(fim - inicio)
    """


OC_CACHE = defaultdict(list)


def _find_winner(free, pf, oc):
    "Returns None if no winner, True if X wins, False if O wins"

    global tsp_obj, C_global, c_global, q_global, p_global, r_global, Q_global

    if len(free) > 0:
        raise RuntimeError(f"find winner call called on nonterminal board {free, pf, oc}")

    if not oc:
        return None

    conjunto_com_ocasionais = []

    for i in oc:
        if random.random() <= p_global[i]:  # oc accepted
            conjunto_com_ocasionais.append(i)

    """
    conjunto_com_ocasionais = []

    for val in oc:
        conjunto_com_ocasionais.append(val)
    """

    z = cache_archetti(C_global, c_global, q_global, r_global, Q_global, conjunto_com_ocasionais)
    OC_CACHE[oc].append(z)

    # print(f"oc={conjunto_com_ocasionais}, z = {z}")

    if z < tsp_obj:
        return True
    elif z > tsp_obj:
        return False
    else:
        return None


def new_lastmile(C):
    free = tuple(C)  # yet to fix
    pf = frozenset()  # professional fleet's vertices
    oc = frozenset()  # outsourced vertices

    return LastMile(free=free, pf=pf, oc=oc, winner=None, terminal=False)


if __name__ == "__main__":
    main()

"""
    A = set()
    while True:
        sys.stdout.flush()
        # print("+++")
        imin = None
        for i in C:
            z = eval_archetti(C, c, q, p, r, Q, A | set([i]))
            if z < zmin:
                zmin = z
                imin = i
        if imin != None:
            A.add(imin)
            print("+{} -> {:.4g}".format(A,zmin))

            while True:
                # print("---")
                jmin = None
                for j in A - set([imin]):
                    z = eval_archetti(C, c, q, p, r, Q, A - set([j]))
                    if z < zmin:
                        zmin = z
                        jmin = j
                if jmin == None:
                    break
                A.remove(jmin)
                print("-{}:{} -> {:.4g}".format(A,jmin,zmin))
        else:
            break

    print("solution:")
    print("{} -> {:.4g}".format(A, zmin))
    sys.stdout.flush()
"""