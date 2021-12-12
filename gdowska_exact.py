import random
from gurobipy import *
from collections import namedtuple, defaultdict
from random import choice
from monte_carlo_tree_search import MCTS, Node

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


def make_data_random(n):
    """make_data_random: prepare all relevant data randomly:
    compute matrix distance based on euclidean distance
    between n customers, m occasional drivers.
    Returns tuple (C, c, q, K, p, alpha, probmax, v0, Q), where:
        - C: list of customers [depot is 0]
        - c[i,j]: cost of edge (i,j)
        - q[i]: demand for customer i
        - p[i]: probability of an outsource to i being accepted
        - r[i]: cost for outsourcing delivery to i
        - Q: vehicle's capacity
      """
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
        r[i] = random.randint(5, 15) # cost for outsourcing delivery to i

    # VEHICLES
    #Q = 50  # vehicle's capacity
    Q = len(C)  # vehicle's capacity

    return C, c, q, p, r, Q, x, y


def make_data_read(n):
    """make_data_read: read all relevant data:
    compute matrix distance based on euclidean distance
    between n customers, m occasional drivers.
    Returns tuple (C, c, q, K, p, alpha, probmax, v0, Q), where:
        - C: list of customers [depot is 0]
        - c[i,j]: cost of edge (i,j)
        - q[i]: demand for customer i
        - p[i]: probability of an outsource to i being accepted
        - r[i]: cost for outsourcing delivery to i
        - Q: vehicle's capacity
      """
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


def main():
    import time
    import itertools
    import sys

    global tsp_obj, C_global, c_global, q_global, p_global, r_global, Q_global

    opcao = input('Deseja ler uma instância (1/0)?\n')

    if opcao == "1":
        n = int(input("Número de Clientes = "))
        C, c, q, p, r, Q, x, y = make_data_read(n)
    else:
        """# initialize the random generator
        try:
            r1 = int(sys.argv[1])
        except:
            r1 = int(time.time())
        print("Random seeds: r1=%d" % (r1))
        random.seed(r1)

        try:
            n = int(sys.argv[2])
        except:
            n = 14  # customers to serve

        C, c, q, p, r, Q, x, y = make_data_random(n)"""

        n = 12
        C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        x = {0: 3, 1: 23, 2: 42, 3: 34, 4: 17, 5: 91, 6: 8, 7: 83, 8: 25, 9: 79, 10: 69, 11: 39, 12: 83}
        y = {0: 15, 1: 78, 2: 53, 3: 8, 4: 47, 5: 78, 6: 12, 7: 53, 8: 82, 9: 58, 10: 35, 11: 72, 12: 10}
        q = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}
        p = {0: 0.26266369929717515, 1: 0.9105203232699572, 2: 0.5761007748204056, 3: 0.9693661729369689,
             4: 0.04330927647637206, 5: 0.4527361806679503, 6: 0.2997500515945183, 7: 0.7681239442857066,
             8: 0.7287134404031639, 9: 0.5117792891431739, 10: 0.97705279241079, 11: 0.20444750927283328,
             12: 0.08331003390462965}
        r = {0: 6, 1: 3, 2: 10, 3: 3, 4: 9, 5: 2, 6: 9, 7: 7, 8: 7, 9: 6, 10: 5, 11: 6, 12: 2}
        Q = 12

        """
        n = 13
        C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        x = {0: 32, 1: 80, 2: 10, 3: 86, 4: 5, 5: 32, 6: 39, 7: 25, 8: 68, 9: 65, 10: 77, 11: 99, 12: 74, 13: 87}
        y = {0: 6, 1: 77, 2: 4, 3: 77, 4: 52, 5: 94, 6: 49, 7: 3, 8: 48, 9: 80, 10: 14, 11: 2, 12: 13, 13: 70}
        q = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1}
        p = {0: 0.020669624592521307, 1: 0.2990840318396233, 2: 0.021501051369915092, 3: 0.4640928902089929,
             4: 0.7862898589578172, 5: 0.26934051004895454, 6: 0.8982378243987975, 7: 0.704443064847012,
             8: 0.8732709690654726, 9: 0.0654237246494167, 10: 0.4010964490810075, 11: 0.07989146742844044,
             12: 0.559246320424422, 13: 0.3826072495437086}
        r = {0: 14, 1: 10, 2: 11, 3: 11, 4: 10, 5: 13, 6: 14, 7: 12, 8: 10, 9: 15, 10: 15, 11: 11, 12: 10, 13: 15}
        Q = 13
        """

        """
        n = 14
        C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        x = {0: 57, 1: 60, 2: 91, 3: 77, 4: 52, 5: 50, 6: 99, 7: 1, 8: 12, 9: 94, 10: 44, 11: 23, 12: 27, 13: 33,
             14: 85}
        y = {0: 3, 1: 52, 2: 13, 3: 56, 4: 6, 5: 8, 6: 38, 7: 81, 8: 31, 9: 45, 10: 16, 11: 78, 12: 100, 13: 61, 14: 28}
        q = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1}
        p = {0: 0.3651823392606387, 1: 0.4295896155160178, 2: 0.818411756528453, 3: 0.1948885488542671,
             4: 0.6277083324079809, 5: 0.27518367917667363, 6: 0.3048153363238888, 7: 0.8066454911376206,
             8: 0.5942252122559062, 9: 0.13593155714825245, 10: 0.6079522100064465, 11: 0.5371995793533364,
             12: 0.9954007884255917, 13: 0.6472665214458186, 14: 0.3190960306606727}
        r = {0: 5, 1: 5, 2: 5, 3: 13, 4: 15, 5: 6, 6: 13, 7: 7, 8: 15, 9: 14, 10: 5, 11: 10, 12: 11, 13: 6, 14: 5}
        Q = 14"""

        distance = lambda x1, y1, x2, y2: int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        c = {}

        for i in [0] + C:
            for j in [0] + C:
                c[i, j] = distance(x[i], y[i], x[j], y[j])
                c[j, i] = distance(x[i], y[i], x[j], y[j])

    C_global = C
    c_global = c
    q_global = q
    p_global = p
    r_global = r
    Q_global = Q

    # solution with no outsourcing
    tsp_obj = cache_archetti(C, c, q, r, Q, [])
    zmin = tsp_obj
    print("SOLUÇÃO SEM ENTREGADORES OCASIONAIS -> {:.4g}".format(tsp_obj))

    tree = MCTS()
    board = new_lastmile(C)

    for _ in range(1000):
        tree.do_rollout(board)

    global OC_CACHE
    #global PATCACHE

    amontecarlo = []
    zmontecarlo = tsp_obj

    for key, value in OC_CACHE.items():
        valor_medio = sum(value) / len(value)

        if valor_medio < zmontecarlo:
            amontecarlo = []

            for val in key:
                amontecarlo.append(val)

            zmontecarlo = valor_medio

    """for key, value in PATCACHE.items():
        if value < zmontecarlo:
            amontecarlo = []

            for val in key:
                amontecarlo.append(val)

            zmontecarlo = value"""

    amontecarlo.sort()
    zmontecarlo = eval_archetti(C, c, q, p, r, Q, amontecarlo)

    print("MIN - MONTE CARLO - OTIMIZADO")
    print("{:.4g} <- {}".format(zmontecarlo, amontecarlo))

    """
    C = []
    C.extend(range(1, n + 1))
    w = len(C)
    masks = [1 << i for i in range(w)]
    for i in range(1 << w):
        A = [ss for mask, ss in zip(masks, C) if i & mask]
        z = eval_archetti(C, c, q, p, r, Q, A)
        #print("{:.4g} <- {}".format(z, A))
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

    return n, x, y, amin, modelAll, modelA"""


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

    """conjunto_com_ocasionais = []

    for val in oc:
        conjunto_com_ocasionais.append(val)"""

    z = cache_archetti(C_global, c_global, q_global, r_global, Q_global, conjunto_com_ocasionais)
    OC_CACHE[oc].append(z)

    print(f"oc={conjunto_com_ocasionais}, z = {z}")

    if z < tsp_obj:
        return True
    elif z > tsp_obj:
        return False
    else:
        return None


def new_lastmile(C):
    free = tuple(C)   # yet to fix
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