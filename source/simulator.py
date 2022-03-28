import Input

BaseLine = 1    # Mbps
LightPathBandwidth = 1000 * BaseLine


if __name__ == '__main__':
    G = Input.InputImp.InputImp().generate_topology(path='../graphml/nsfnet/nsfnet.graphml')
    Input.InputImp.InputImp().generate_adjacency_martix(G)
    Input.InputImp.InputImp().generate_traffic_matrix(nodes=[1, 2, 3])
    pass