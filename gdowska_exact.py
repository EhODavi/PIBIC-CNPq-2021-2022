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
        r[i] = random.randint(5, 10) # cost for outsourcing delivery to i

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
        """
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

        """
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
        Q = 14
        """

        """
        n = 50
        C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
        x = {0: 51, 1: 95, 2: 45, 3: 22, 4: 16, 5: 76, 6: 86, 7: 97, 8: 1, 9: 16, 10: 40, 11: 79, 12: 80, 13: 99,
             14: 49, 15: 82, 16: 63, 17: 11, 18: 26, 19: 12, 20: 85, 21: 66, 22: 42, 23: 48, 24: 63, 25: 54, 26: 92,
             27: 76, 28: 40, 29: 64, 30: 69, 31: 6, 32: 100, 33: 4, 34: 26, 35: 64, 36: 59, 37: 82, 38: 67, 39: 48,
             40: 62, 41: 89, 42: 99, 43: 48, 44: 34, 45: 33, 46: 67, 47: 10, 48: 18, 49: 13, 50: 71}
        y = {0: 7, 1: 99, 2: 10, 3: 71, 4: 13, 5: 76, 6: 80, 7: 74, 8: 54, 9: 38, 10: 20, 11: 100, 12: 39, 13: 25,
             14: 88, 15: 64, 16: 88, 17: 61, 18: 64, 19: 27, 20: 53, 21: 90, 22: 39, 23: 37, 24: 24, 25: 55, 26: 87,
             27: 77, 28: 46, 29: 94, 30: 42, 31: 64, 32: 95, 33: 25, 34: 71, 35: 2, 36: 47, 37: 78, 38: 31, 39: 13,
             40: 66, 41: 19, 42: 2, 43: 7, 44: 52, 45: 99, 46: 15, 47: 69, 48: 25, 49: 51, 50: 35}
        q = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1,
             16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1,
             31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1, 40: 1, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1,
             46: 1, 47: 1, 48: 1, 49: 1, 50: 1}
        p = {0: 0.7795767662728305, 1: 0.9329854499432126, 2: 0.5334193085107671, 3: 0.3772506683982553,
             4: 0.5592048863782512, 5: 0.1584985854661427, 6: 0.3277387615431181, 7: 0.5745472913327511,
             8: 0.7654343486059647, 9: 0.5452907534442316, 10: 0.3051290493972607, 11: 0.4014310206330801,
             12: 0.036408526321180634, 13: 0.21993130578757314, 14: 0.2626923689446867, 15: 0.8859658955606651,
             16: 0.4651541930212977, 17: 0.480739798290014, 18: 0.8816751273644882, 19: 0.41272270358261787,
             20: 0.7055510508347221, 21: 0.9707542549006581, 22: 0.6036582494280214, 23: 0.35517300610524183,
             24: 0.23648957760097367, 25: 0.4210068451640976, 26: 0.3759655940243918, 27: 0.9819689143531202,
             28: 0.5630146678433725, 29: 0.17156086695951167, 30: 0.6976427765174451, 31: 0.006888369334165367,
             32: 0.019132482145583496, 33: 0.05426342950219076, 34: 0.7768026210725316, 35: 0.1801919180969339,
             36: 0.2503338923993351, 37: 0.07596322812127287, 38: 0.17160104015984012, 39: 0.42674413474819306,
             40: 0.11852127636767773, 41: 0.17360211505695566, 42: 0.07995672999150094, 43: 0.5534036560961713,
             44: 0.4250635736078199, 45: 0.0960997549212278, 46: 0.4684652677780484, 47: 0.9633712350497492,
             48: 0.19528276747824913, 49: 0.49083586608247387, 50: 0.5524670176293808}
        r = {0: 10, 1: 6, 2: 7, 3: 7, 4: 7, 5: 9, 6: 9, 7: 8, 8: 7, 9: 7, 10: 9, 11: 5, 12: 7, 13: 7, 14: 10, 15: 6,
             16: 10, 17: 5, 18: 5, 19: 10, 20: 6, 21: 9, 22: 6, 23: 9, 24: 6, 25: 9, 26: 6, 27: 7, 28: 7, 29: 10, 30: 8,
             31: 10, 32: 8, 33: 7, 34: 5, 35: 8, 36: 9, 37: 9, 38: 9, 39: 6, 40: 10, 41: 10, 42: 9, 43: 5, 44: 7, 45: 5,
             46: 6, 47: 9, 48: 9, 49: 10, 50: 6}
        Q = 50
        """

        """
        n = 12
        C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        x = {0: 3, 1: 23, 2: 42, 3: 34, 4: 17, 5: 91, 6: 8, 7: 83, 8: 25, 9: 79, 10: 69, 11: 39, 12: 83}
        y = {0: 15, 1: 78, 2: 53, 3: 8, 4: 47, 5: 78, 6: 12, 7: 53, 8: 82, 9: 58, 10: 35, 11: 72, 12: 10}
        q = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}
        p = {0: 0.8, 1: 0.85, 2: 0.81, 3: 0.83, 4: 0.99, 5: 0.94, 6: 0.91, 7: 0.92, 8: 0.89, 9: 0.9, 10: 0.91, 11: 0.98,
             12: 0.91}
        r = {0: 6, 1: 3, 2: 10, 3: 3, 4: 9, 5: 2, 6: 9, 7: 7, 8: 7, 9: 6, 10: 5, 11: 6, 12: 2}
        Q = 12
        """

        """
        n = 12
        C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        x = {0: 3, 1: 23, 2: 42, 3: 34, 4: 17, 5: 91, 6: 8, 7: 83, 8: 25, 9: 79, 10: 69, 11: 39, 12: 83}
        y = {0: 15, 1: 78, 2: 53, 3: 8, 4: 47, 5: 78, 6: 12, 7: 53, 8: 82, 9: 58, 10: 35, 11: 72, 12: 10}
        q = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}
        p = {0: 0.1, 1: 0.05, 2: 0.15, 3: 0.19, 4: 0.15, 5: 0.12, 6: 0.13, 7: 0.14, 8: 0.15, 9: 0.06, 10: 0.07,
             11: 0.08, 12: 0.13}
        r = {0: 6, 1: 3, 2: 10, 3: 3, 4: 9, 5: 2, 6: 9, 7: 7, 8: 7, 9: 6, 10: 5, 11: 6, 12: 2}
        Q = 12
        """

        """
        n = 13
        C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        x = {0: 32, 1: 80, 2: 10, 3: 86, 4: 5, 5: 32, 6: 39, 7: 25, 8: 68, 9: 65, 10: 77, 11: 99, 12: 74, 13: 87}
        y = {0: 6, 1: 77, 2: 4, 3: 77, 4: 52, 5: 94, 6: 49, 7: 3, 8: 48, 9: 80, 10: 14, 11: 2, 12: 13, 13: 70}
        q = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1}
        p = {0: 0.81, 1: 0.95, 2: 0.94, 3: 0.99, 4: 0.92, 5: 0.88, 6: 0.89, 7: 0.91, 8: 0.96, 9: 0.92, 10: 0.92,
             11: 0.91, 12: 0.84, 13: 0.82}
        r = {0: 14, 1: 10, 2: 11, 3: 11, 4: 10, 5: 13, 6: 14, 7: 12, 8: 10, 9: 15, 10: 15, 11: 11, 12: 10, 13: 15}
        Q = 13
        """

        """
        n = 13
        C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        x = {0: 32, 1: 80, 2: 10, 3: 86, 4: 5, 5: 32, 6: 39, 7: 25, 8: 68, 9: 65, 10: 77, 11: 99, 12: 74, 13: 87}
        y = {0: 6, 1: 77, 2: 4, 3: 77, 4: 52, 5: 94, 6: 49, 7: 3, 8: 48, 9: 80, 10: 14, 11: 2, 12: 13, 13: 70}
        q = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1}
        p = {0: 0.05, 1: 0.06, 2: 0.01, 3: 0.02, 4: 0.04, 5: 0.10, 6: 0.11, 7: 0.15, 8: 0.12, 9: 0.13, 10: 0.14,
             11: 0.09, 12: 0.05, 13: 0.04}
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
        p = {0: 0.94, 1: 0.93, 2: 0.91, 3: 0.99, 4: 0.92, 5: 0.92, 6: 0.95, 7: 0.99, 8: 0.92, 9: 0.91, 10: 0.9,
             11: 0.88, 12: 0.91, 13: 0.91, 14: 0.95}
        r = {0: 5, 1: 5, 2: 5, 3: 13, 4: 15, 5: 6, 6: 13, 7: 7, 8: 15, 9: 14, 10: 5, 11: 10, 12: 11, 13: 6, 14: 5}
        Q = 14
        """

        """
        n = 14
        C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        x = {0: 57, 1: 60, 2: 91, 3: 77, 4: 52, 5: 50, 6: 99, 7: 1, 8: 12, 9: 94, 10: 44, 11: 23, 12: 27, 13: 33,
             14: 85}
        y = {0: 3, 1: 52, 2: 13, 3: 56, 4: 6, 5: 8, 6: 38, 7: 81, 8: 31, 9: 45, 10: 16, 11: 78, 12: 100, 13: 61, 14: 28}
        q = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1}
        p = {0: 0.05, 1: 0.01, 2: 0.04, 3: 0.1, 4: 0.09, 5: 0.08, 6: 0.05, 7: 0.03, 8: 0.03, 9: 0.02, 10: 0.04,
             11: 0.05, 12: 0.06, 13: 0.02, 14: 0.01}
        r = {0: 5, 1: 5, 2: 5, 3: 13, 4: 15, 5: 6, 6: 13, 7: 7, 8: 15, 9: 14, 10: 5, 11: 10, 12: 11, 13: 6, 14: 5}
        Q = 14
        """

        """
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
        Q = 4
        """

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
        Q = 5
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
        Q = 5
        """

        distance = lambda x1, y1, x2, y2: int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        c = {}

        for i in [0] + C:
            for j in [0] + C:
                c[i, j] = distance(x[i], y[i], x[j], y[j])
                c[j, i] = distance(x[i], y[i], x[j], y[j])

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
    zmin = tsp_obj
    amin = []
    print("SOLUÇÃO SEM ENTREGADORES OCASIONAIS -> {:.4g}".format(tsp_obj))

    tree = MCTS()
    board = new_lastmile(C)

    for _ in range(10000):
        tree.do_rollout(board)

    #global OC_CACHE
    global PATCACHE

    amontecarlo = []
    zmontecarlo = tsp_obj

    """
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

    amontecarlo.sort()
    #zmontecarlo = eval_archetti(C, c, q, p, r, Q, amontecarlo)

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

    return n, x, y, amin, modelAll, modelA
    """


#OC_CACHE = defaultdict(list)


def _find_winner(free, pf, oc):
    "Returns None if no winner, True if X wins, False if O wins"

    global tsp_obj, C_global, c_global, q_global, p_global, r_global, Q_global

    if len(free) > 0:
        raise RuntimeError(f"find winner call called on nonterminal board {free, pf, oc}")

    if not oc:
        return None

    """
    conjunto_com_ocasionais = []

    for i in oc:
        if random.random() <= p_global[i]:  # oc accepted
            conjunto_com_ocasionais.append(i)
    """

    conjunto_com_ocasionais = []

    for val in oc:
        conjunto_com_ocasionais.append(val)

    z = eval_archetti(C_global, c_global, q_global, p_global, r_global, Q_global, conjunto_com_ocasionais)
    #OC_CACHE[oc].append(z)

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