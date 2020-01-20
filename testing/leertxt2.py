# nombre  /  anomalo/ ini  / fin


class normal():
    def __init__(self, n, t):
        self.name = n
        self.type = t


class anomalous(normal):
    def __init__(self, n, t, i, f):
        normal.__init__(self, n, t)
        self.start = i
        self.end = f

def leer(src_normal, src_anomalous):
    lista1, lista2 = [], []
    arch1 = open(src_normal, 'r')
    for line in arch1:
        line=line[:-1]
        v_n = normal(line, 0)
        lista1.append(v_n)
    arch1.close()
    arch2 = open(src_anomalous, 'r')
    for line in arch2:
        linea = line.split(' ')
        v_n = anomalous(linea[0], 1, linea[1], linea[2])
        lista2.append(v_n)
    arch1.close()
    return lista1, lista2


#l1, l2 = leer("Normales.txt", "Anomalous.txt")
""" for i in l1:
    print(vars(i))
for i in l2:
    print(vars(i)) """
