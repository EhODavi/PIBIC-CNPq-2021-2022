import random
import math
from gurobipy import *


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


def make_data(n):
    """make_data: prepare all relevant data randomly:
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
    x = {i:random.random() for i in [0]+C}
    y = {i:random.random() for i in [0]+C}
    c, q = {}, {}  # c[i,j] -> cost matrix, q[i] -> demand
    p, r = {}, {}  # p[i], r[i] -> prob/cost for outsourcing to i
    beta = {}      # beta[i,k] = 1 if i can be served by k
    for i in [0]+C:
        q[i] = random.randint(10,20)  # demand for customer i
        for j in [0]+C:
            c[i,j] = distance(x[i],y[i],x[j],y[j])
            c[j,i] = distance(x[i],y[i],x[j],y[j])

    # COURIERS
    for i in [0] + C:
        p[i] = random.random()  # probability of an outsource to i being accepted
        r[i] = random.random()  # cost for outsourcing delivery to i

    # VEHICLES
    Q = 50  # vehicle's capacity

    return C, c, q, p, r, Q



if __name__ == "__main__":
    import time
    import itertools
    import sys

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
    	n = 15   # customers to serve
    C, c, q, p, r, Q = make_data(n)

    # solution with no outsourcing
    amin = []
    zmin = cache_archetti(C, c, q, r, Q, [])
    print("initial -> {:.4g}".format(zmin))

    C = []
    C.extend(range(1, n+1))
    x = len(C)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
	    A = [ss for mask, ss in zip(masks, C) if i & mask]
	    z = eval_archetti(C, c, q, p, r, Q, A)
	    print("{:.4g} <- {}".format(z,A))
	    if z<zmin:
		    zmin = z
		    amin = A
    print("MIN")
    print("{:.4g} <- {}".format(zmin,amin))
    sys.stdout.flush()

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