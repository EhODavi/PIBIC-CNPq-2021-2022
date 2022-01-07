import sys
import os

if len(sys.argv) < 3:
    print(sys.argv[0] + " #simulations compensation [compensation2 …]")
    exit()

for c in sys.argv[2:]:
    os.system("echo \"** Comp = " + c + " " + c + "\" | tee >> saida.txt")

    for i in range(0, int(sys.argv[1])):
        os.system("python3 s-lmdloc.py " + str(i) + " " + c + " " + c + " >> saida.txt")
