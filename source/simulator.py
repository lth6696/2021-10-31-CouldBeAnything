import Input


if __name__ == '__main__':
    G = Input.InputImp.InputImp().generate_topology(path='../topology/nsfnet/nsfnet.graphml')
    Input.InputImp.InputImp().generate_adjacency_martix(G)
    Input.InputApi.Input().generate_traffic_matrix(2)
    pass