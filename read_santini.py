import gzip
import math


def read_santini(filename):
    """basic function for reading an instance in the TSPLIB+Santini format
        :returns: V, c, x, y, prob, comp
        where:
            V: set of vertices
            c[i,j]: cost of moving from vertex i to j
            x[i], y[i]: node i coordinates
            prob[i], comp[i]: probability of acceptance and compensation for i
        NOTE: implemented only EUC_2D format"""

    if filename[-3:] == ".gz":
        f = gzip.open(filename, 'rt')
    else:
        f = open(filename)

    line = f.readline()
    while line.find("DIMENSION") == -1:
        line = f.readline()
    n = int(line.split()[-1])

    while line.find("EDGE_WEIGHT_TYPE") == -1:
        line = f.readline()

    if line.find("EUC_2D") != -1:
        dist = lambda x1, y1, x2, y2: int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    else:
        print("cannot deal with '%s' distances" % line)
        raise Exception

    while line.find("NODE_COORD_SECTION") == -1:
        line = f.readline()

    V = []  # jpp: changed on 2019-02-08 for having an order on vertices (before V was x.keys())
    x, y = {}, {}
    while 1:
        line = f.readline()
        if line.find("ACCEPTED_PROBABILITIES") != -1: break
        (i, xi, yi) = line.split()
        V.append(i)
        x[i] = float(xi)
        y[i] = float(yi)

    c = {}  # dictionary to hold n times n matrix
    for i in V:
        for j in V:
            c[i, j] = dist(x[i], y[i], x[j], y[j])

    prob = {}
    for i in V:
        line = f.readline()
        prob[i] = float(line)

    line = f.readline()
    assert ("OUTSOURCING_COSTS" in line)
    comp = {}
    for i in V:
        line = f.readline()
        comp[i] = float(line)

    f.close()

    return V, c, x, y, prob, comp


if __name__ == "__main__":
    filename = "ptspc-instances-master/instances/sz-10-prob_type-direct_dist-prob-0.25-fee_type-direct_prob-fee-2.5.txt.gz"
    print(f"reading {filename}...")
    V, c, x, y, prob, comp = read_santini(filename)
    print(f"read {len(V)} nodes")
    for i in V:
        print(i, prob[i], comp[i])
