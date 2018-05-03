import sys
import random

verts = 0
max_x = 50
max_y = 20
max_z = 50
max_r = 10

def main():
    global max_x, max_y, max_z, max_r
    args = sys.argv[1:]
    geomtype = "t"
    n = 1

    try:
        assert len(args) >= 2

        geomtype = str(args[0])
        assert len(geomtype) == 1

        n = int(args[1])

        max_x = float(args[2])
        max_y = float(args[3])
        max_z = float(args[4])

        if len(args) >= 6:
            max_r = float(args[5])

    except:
        print("Usage: generator.py [type geom (s or t)] [num geoms] [max_x] [max_y] [max_z] [max_r]", file=sys.stderr)
        quit()

    for _ in range(n):
        if geomtype == "s":
            make_spheres()
        elif geomtype == "t":
            make_tris()

    for i in range(0, verts, 3):
        print("f " + str(i+1) + " " + str(i+2) + " " + str(i+3))

def make_spheres():
    x = (random.random() * 2 * max_x) - max_x
    y = (random.random() * 2 * max_y) - max_y
    z = (random.random() * 2 * max_z) - max_z
    r = (random.random() * 2 * max_r) - max_r

    print("sphere", x, y, z, r)

def make_tris():
    global verts
    
    x = (random.random() * 2 * max_x) - max_x
    y = (random.random() * 2 * max_y) - max_y
    z = (random.random() * 2 * max_z) - max_z
    for _ in range(3):
        dx = random.uniform(-max_r, max_r)
        dy = random.uniform(-max_r, max_r)
        dz = random.uniform(-max_r, max_r)

        verts+=1
        print("v", x + dx, y + dy, z + dz)


main()
