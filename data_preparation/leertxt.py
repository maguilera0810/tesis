# nombre  /  anomalo/ ini  / fin


import rutas_data_preparation as rt


class normal():
    def __init__(self, nombre, tipo):
        self.name = nombre
        self.type = tipo


class section():
    def __init__(self, inicio, duracion):
        self.inicio = inicio
        self.duracion = duracion


class anomalous(normal):
    def __init__(self, nombre, tipo, listaTramosAnomalos=[], listaTramosNoUtiles=[]):
        normal.__init__(self, nombre, tipo)
        self.tramosAnomalos = listaTramosAnomalos
        self.tramos_no_usar = listaTramosNoUtiles


def leer(src_normal, src_anomalous):
    lista1, lista2 = [], []
    arch1 = open(src_normal, 'r')
    for line in arch1:
        lista1.append(
            normal(line.rstrip('\n'),0)
        )
    arch1.close()
    arch2 = open(src_anomalous, 'r')
    for line in arch2:
        linea = line.rstrip('\n').split("/")

        linea_no_usar = linea[1].split(" ")
        tramosNoUsar = []
        i = 0
        while i < len(linea_no_usar)-1:
            tramosNoUsar.append(
                section(linea_no_usar[i], linea_no_usar[i+1]))
            i += 2

        linea_anomalos = linea[2].split(" ")
        # print(linea_anomalos)
        tramosAnomalos = []
        i = 0
        while i < len(linea_anomalos)-1:
            tramosAnomalos.append(
                section(linea_anomalos[i], linea_anomalos[i+1]))
            i += 2
        lista2.append(
            anomalous(
                nombre=linea[0],
                tipo=1,
                listaTramosAnomalos=tramosAnomalos,
                listaTramosNoUtiles=tramosNoUsar
            )
        )
    arch1.close()
    return lista1, lista2


if __name__ == '__main__':
    paths = rt.Directorios("..")
    print(paths.normal_training_data_txt)
    print(paths.anomalous_training_data_txt)
    l1, l2 = leer(paths.normal_training_data_txt,
                  paths.anomalous_training_data_txt)

    for i in l2:
        print(vars(i))
    for i in l1:
        print(vars(i))
    """     print(i.name, i.type, end=" ")
        print("-", end="")
        for t in i.tramos_no_usar:
            print(t.inicio, t.duracion, end=" ")
        print("-", end="")
        for t in i.tramosAnomalos:
            print(t.inicio, t.duracion, end=" ")
        print("") """
