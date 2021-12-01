"""
An example implementation of the abstract Node class for use in MCTS
"""

from collections import namedtuple, defaultdict, Counter
from random import choice
from monte_carlo_tree_search import MCTS, Node

_Gdowska = namedtuple("Gdowska", "free pf oc winner terminal")
c = {}   # cannot be an attribute of _Gdowska...  !!!
tsp_obj = None

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
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


def play_game(V, x, y, prob, comp, N):
    global tsp_obj
    print(f"at root, {len(V)} variables to choose from")
    tsp_obj,edges = solve_tsp(V,c)
    seq = sequence(list(sorted(V)),edges)
    succ = {seq[i-1]:seq[i] for i in range(len(seq))}
    prev = {seq[i]:seq[i-1] for i in range(len(seq))}
    print("full tsp:", tsp_obj, seq)
    print("    succ:", succ)
    print("    prev:", prev)
    savings = {i:((c[prev[i],i] + c[i,succ[i]]) -
                  (prob[i]*(c[prev[i],succ[i]] - c[prev[i],i] - c[i,succ[i]] + comp[i]) +
                  (1 - prob[i])*(c[prev[i],i] + c[i,succ[i]]))) for i in sorted(V[1:])}   # !!! simplify!
    print("savings:", savings)
    V = V[0:1] + list(sorted(V[1:], key=lambda k: savings[k], reverse=False))
    print("sorted V:", V)

    tree = MCTS()
    board = new_lastmile(V)
    print("initial board:", board.to_pretty_string())
    for _ in range(N):
        tree.do_rollout(board)
    while not board.is_terminal():
        board = tree.choose(board)
        print("chose board:", board.to_pretty_string())
    print(board.oc)
    print(OC_CACHE[board.oc])
    print(sum(OC_CACHE[board.oc])/len(OC_CACHE[board.oc]))
    for key in sorted(OC_CACHE, key=lambda k: sum(OC_CACHE[k])/len(OC_CACHE[k]), reverse=False):
        print(f"{key}\t{Counter(OC_CACHE[key])}\t{sum(OC_CACHE[key])/len(OC_CACHE[key])}\t{len(OC_CACHE[key])}")


OC_CACHE = defaultdict(list)
def _find_winner(free, pf, oc):
    "Returns None if no winner, True if X wins, False if O wins"
    global tsp_obj
    print("_find_winner", free, pf, oc, end="\t")
    if len(free) > 0:
        raise RuntimeError(f"find winner call called on nonterminal board {free, pf, oc}")

    if not oc:
        print(None)
        return None
    comp_total = 0
    pf = set(pf)
    for i in oc:
        if random.random() <= prob[i]:  # oc accepted
            comp_total += comp[i]
        else:
            pf.add(i)

    obj = cache_tsp(pf, c)
    OC_CACHE[oc].append(obj)
    if comp_total + obj < tsp_obj:
        print(comp_total + obj, True)
        return True
    elif comp_total + obj > tsp_obj:
        print(comp_total + obj, False)
        return False
    else:
        print(comp_total + obj, None)
        return None

def new_lastmile(V):
    free = tuple(V[1:])   # yet to fix
    pf = frozenset(V[0])  # professional fleet's vertices
    oc = frozenset()  # outsourced vertices
    return LastMile(free=free, pf=pf, oc=oc, winner=None, terminal=False)


from read_santini import read_santini
from tsp import solve_tsp, sequence

# globals
EPS = 1.e-9
MIPCACHE = {}
def cache_tsp(V, c):
    """
    check if current tsp model has already been solved; if not, solve and chache it
    Parameters: solve_tsp
        - V: set of customers to visit
        - c[i,j]: cost of traversing edge from customer i to j
    Returns the optimum.
    """
    key = frozenset(V)
    if key in MIPCACHE:
        return MIPCACHE[key]

    if len(V) <= 1:
        obj = 0
        edges = []
    elif len(V) == 2:
        i = V[0]
        j = V[1]
        obj = c[i, j] + c[j, i]
        edges = [(i, j)]
    else:
        obj,edges = solve_tsp(V,c)

    # print(V, "-> Optimum TSP:", obj)
    # if len(V) > 1:
    #     print(sequence(list(sorted(V)),edges), " ::: ", sum(c[i,j] for (i,j) in edges))
    #     for (i,j) in edges:
    #         print("\t", (i,j), ":", c[i,j])
    # # print("Optimal cost:", obj, "+", fix)

    MIPCACHE[key] = obj
    return MIPCACHE[key]


if __name__ == "__main__":
    import sys
    import random
    # random.seed(0)

    if len(sys.argv) != 3:
        print("usage: {} filename N".format(sys.argv[0]))
        print("    where")
        print("      - filename is the instance file (Santini's format)")
        print("      - N is the number of iterations")
        exit(-1)

    V, c_, x, y, prob, comp = read_santini(sys.argv[1])
    # !!!
    for key in c_:
        c[key] = c_[key]
    # !!!
    play_game(V, x, y, prob, comp, int(sys.argv[2]))
