import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import pandas as pd


pd.set_option('display.max_column', 200)
pd.set_option('display.width', 200)


class Traffic(object):
    def __init__(self, src, dst, resource, t):
        self.src = src
        self.dst = dst
        self.resource = resource
        self.data = None
        self.key = None
        self.t = t
        self.path = []

        self.__cal_key()

    def __cal_key(self):
        self.data = self.resource * 1 / (self.t + 1)
        self.key = self.resource * self.t / (self.t + 1)


class LightPath(object):
    def __init__(self, start, end, index, resource, t):
        self.start = start
        self.end = end
        self.index = index
        self.resource = resource
        self.ava_data = 0
        self.ava_key = 0
        self.t = t

        self.__cal_ava_resource()

    def __cal_ava_resource(self):
        self.ava_data = self.resource * 1 / (self.t + 1)
        self.ava_key = self.resource * self.t / (self.t + 1)


# generate traffic matrix
def gen_traffic_matrix(nodes, tafs, max_res):
    random.seed(0)
    traffic_matrix = [[None for _ in range(nodes)] for _ in range(nodes)]
    for src in range(nodes):
        for dst in range(nodes):
            if src != dst:
                traffic_matrix[src][dst] = Traffic(src,
                                                   dst,
                                                   random.randint(0, max_res),
                                                   tafs[random.randint(0, len(tafs)-1)]
                                                   )
    return traffic_matrix


# generate lightpath topology
def gen_lightpath_topo(path, wavelengths, tafs, max_res):
    random.seed(0)
    G = nx.Graph(nx.read_graphml(path))

    adj_matrix = np.array(nx.adjacency_matrix(G).todense())
    adj_matrix_with_lightpath = [[[] for _ in range(len(G.nodes))] for _ in range(len(G.nodes))]
    for row in range(len(adj_matrix)):
        for col in range(len(adj_matrix)):
            adj_matrix[row][col] = adj_matrix[row][col] * random.randint(1, wavelengths)
            for t in range(adj_matrix[row][col]):
                lightpath = LightPath(row, col, t, max_res, tafs[random.randint(0, len(tafs)-1)])
                adj_matrix_with_lightpath[row][col].append(lightpath)

    return adj_matrix_with_lightpath


# A heuristic algorithm
def heuristic_algorithm(max_hop, traffic_load, average_hops=False, res_utilization=False):
    root = '../datasets/nsfnet/nsfnet.graphml'
    nodes = 14
    tafs = [0.001, 0.0001, 0.0005, 0.00001]
    lightpath_res = 10
    traffic_res = int(lightpath_res * traffic_load)

    success_alloc_matrix = [[0 for _ in range(nodes)] for _ in range(nodes)]
    traffic_matrix = gen_traffic_matrix(nodes, tafs, max_res=traffic_res)
    adj_matrix_with_lightpath = gen_lightpath_topo(root, wavelengths=4, tafs=tafs, max_res=lightpath_res)

    average_hop = []
    for src in range(len(traffic_matrix)):
        for dst in range(len(traffic_matrix)):
            traffic = traffic_matrix[src][dst]
            # 若流量矩阵从src至dst存在连接请求，为其路由并分配资源
            if traffic:
                # 路由
                paths = function_multi_hop(max_hop, src, traffic, adj_matrix_with_lightpath)
                # success_alloc_matrix[src][dst]=1 if successfully routed
                success_alloc_matrix[src][dst] = min(1, len(paths))
                average_hop.append(len(paths))
                # Map traffic resources into paths
                for path in paths:
                    adj_matrix_with_lightpath[path[0]][path[1]][path[2]].ava_data -= traffic.data
                    adj_matrix_with_lightpath[path[0]][path[1]][path[2]].ava_key -= traffic.key

    if average_hops:
        return sum(average_hop) / len(average_hop)
    elif res_utilization:
        data = []
        key = []
        for src in range(len(adj_matrix_with_lightpath)):
            for dst in range(len(adj_matrix_with_lightpath)):
                for edge in adj_matrix_with_lightpath[dst][src]:
                    data.append(1 - edge.ava_data / (edge.resource * 1 / (edge.t + 1)))
                    key.append(1 - edge.ava_key / (edge.resource * edge.t / (edge.t + 1)))
        return np.mean(data), np.mean(key)
    else:
        return success_alloc_matrix


# 基于多约束的最短路径算法，约束有：
# 1 - 安全等级
# 2 - 资源多寡
def function_multi_hop(max_hop, src, traffic, adj_matrix_with_lightpath):
    dst = traffic.dst

    # 设定迭代深度
    if max_hop <= 0:
        return []

    # 判断本次目的地是否可达
    for lightpath in adj_matrix_with_lightpath[src][dst]:
        # if lightpath has enough data and secret-key resources and meets the level requirements
        if lightpath.t >= traffic.t and \
                lightpath.ava_data >= traffic.data and \
                lightpath.ava_key >= traffic.data * traffic.t:
            path = [(src, dst, lightpath.index)]
            return path

    # 若目的地不可直达
    path = []
    path_len = 10
    for j in range(len(adj_matrix_with_lightpath[src])):
        if j != dst:
            for lightpath in adj_matrix_with_lightpath[src][j]:
                # if lightpath has enough data and secret-key resources and meets the level requirements
                if lightpath.t >= traffic.t and \
                        lightpath.ava_data >= traffic.data and \
                        lightpath.ava_key >= traffic.data * traffic.t:
                    prv_hops = function_multi_hop(max_hop-1, j, traffic, adj_matrix_with_lightpath)
                    if len(prv_hops) < path_len and len(prv_hops) > 0:
                        prv_hops.insert(0, (src, j, lightpath.index))
                        path_len = len(prv_hops)
                        path = prv_hops
    return path


def test_runtime():
    import time
    from scipy.interpolate import make_interp_spline

    hops = [2, 3, 4, 5, 6, 7]
    # runtime = []
    # for max_hop in hops:
    #     print("----------current max hop is {}-----------".format(max_hop))
    #     starttime = time.time()
    #     heuristic_algorithm(max_hop, res_utilization=0.5)
    #     endtime = time.time()
    #     runtime.append((endtime - starttime) * 1000)
    # print(runtime)
    runtime = [6.981611251831055, 11.967897415161133, 28.922557830810547,
               104.71987724304199, 543.5464382171631, 2902.683973312378]
    runtime = [i/1000 for i in runtime]
    model = make_interp_spline(hops, runtime)
    xs = np.linspace(2, 7, 500)
    ys = model(xs)
    plot_style()
    plt.plot(xs, ys, lw=0.5)
    plt.plot(hops, runtime, marker='o', linestyle='', ms=2, mew=0.5)
    plt.xlabel('Maximum hops')
    plt.ylabel('Running time (s)')
    plt.yticks(rotation='vertical')
    plt.tight_layout()
    plt.show()


def test_success_route():
    traffic_load = [0.1 * (i+1) for i in range(10)]
    success_route_rate = []
    # for i in traffic_load:
    #     success_alloc_matrix = heuristic_algorithm(max_hop=4, traffic_load=i)
    #     success_alloc_num = sum([sum(i) for i in success_alloc_matrix])
    #     traffic_num = len(success_alloc_matrix) ** 2 - len(success_alloc_matrix)
    #     success_route_rate.append(success_alloc_num / traffic_num * 100)
    # print(success_route_rate)
    success_route_rate = [70.32967032967034, 66.48351648351648, 66.48351648351648, 62.637362637362635, 53.84615384615385,
                          51.098901098901095, 48.35164835164835, 46.15384615384615, 40.10989010989011, 38.46153846153847]
    plot_style()
    plt.plot(traffic_load, success_route_rate, marker='o', ls='-', lw=0.5, ms=2, mew=0.5, mfc='#FB711E', mec='#F77D78')
    plt.xlabel('Traffic load')
    plt.ylabel('Success allocation rate (%)')
    plt.xlim((0, 1.1))
    plt.yticks(rotation='vertical')
    plt.tight_layout()
    plt.show()


def test_average_hop():
    average_hop = {}
    traffic_load = [0.1 * (i + 1) for i in range(10)]
    hops = [2, 3, 4, 5]

    # for max_hop in hops:
    #     tmp = []
    #     for i in traffic_load:
    #         tmp.append(heuristic_algorithm(max_hop=max_hop, traffic_load=i, average_hops=True))
    #     average_hop[max_hop] = tmp

    average_hop = {2: [0.6868131868131868, 0.6813186813186813, 0.6868131868131868, 0.6593406593406593, 0.5934065934065934,
                       0.6263736263736264, 0.6263736263736264, 0.5659340659340659, 0.521978021978022, 0.5],
                   3: [1.3296703296703296, 1.3186813186813187, 1.2417582417582418, 1.2197802197802199, 1.0274725274725274,
                       1.0, 0.9175824175824175, 0.7747252747252747, 0.8186813186813187, 0.7857142857142857],
                   4: [1.5274725274725274, 1.4945054945054945, 1.445054945054945, 1.3956043956043955, 1.2417582417582418,
                       1.120879120879121, 1.0054945054945055, 1.0164835164835164, 0.8846153846153846, 0.8406593406593407],
                   5: [1.6373626373626373, 1.7142857142857142, 1.532967032967033, 1.4945054945054945, 1.3571428571428572,
                       1.2307692307692308, 1.0604395604395604, 0.9615384615384616, 1.0, 0.945054945054945]}

    color = ['#0785D7', '#7F5605', '#FC721E', '#B8B71F']
    mfc = ['#7C9DE9', '#9798B2', '#F7AF6B', '#80DDD6']
    plot_style()
    fig, ax = plt.subplots()  # 创建图实例
    for i, max_hop in enumerate(hops):
        ax.plot(traffic_load, average_hop[max_hop], '.-', label='MH={}'.format(max_hop),
                c=color[i], lw=0.5, mfc=mfc[i], mec='#35526A', ms=3, mew=0.5)
    plt.xlabel('Traffic load')
    plt.ylabel('Average hops')
    plt.xlim((0, 1.1))
    plt.ylim((0.4, 2.4))
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    plt.yticks(rotation='vertical')
    plt.tight_layout()
    plt.show()


def test_res_utilization():
    plot_style()

    # res = []
    traffic_load = [0.1 * (i + 1) for i in range(10)]
    # for i in traffic_load:
    #     res.append(heuristic_algorithm(max_hop=6, traffic_load=i, res_utilization=True))

    res = [(0.17944358912601324, 0.09328803298713124), (0.36196342768874135, 0.20712010053332539),
           (0.4547594174382048, 0.26626539963772083), (0.5063313151549218, 0.2749451390055079),
           (0.5682228558293965, 0.29088958344586324), (0.49188745017243923, 0.2804387485156422),
           (0.5444952804837443, 0.2973926614664543), (0.5630610139008015, 0.2919777197011902),
           (0.5785423971419417, 0.29431627019052187), (0.5424551986824114, 0.27655601640278404)]

    fig, ax = plt.subplots()
    ax.bar(traffic_load, [i[0] for i in res], width=0.05, color='#00a8e1', ec='#3b6291', label='data')
    ax.bar(traffic_load, [i[1] for i in res], width=0.05, color='#99cc00', ec='#779043', label='key')
    ax.legend()
    plt.xlim((0, 1.1))
    plt.xlabel('Traffic load')
    plt.ylabel('Resource utilization')
    plt.yticks(rotation='vertical')
    plt.tight_layout()
    plt.show()

def plot_style():
    plt.rcParams['figure.figsize'] = (1.67, 1.25)  # figure size in inches
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 6
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['patch.linewidth'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'


if __name__ == '__main__':
    # heuristic_algorithm(4, 0.5)
    data = [(2.1868131868131866, 2.1868131868131866, 2.1868131868131866),
            (2.2142857142857144, 2.251990231990232, 2.311111111111111),
            (2.3251533742331287, 2.4262166076606375, 2.540880503144654),
            (2.406015037593985, 2.462477902917937, 2.528985507246377)]
    file = 'shf_hop.mat'
    import scipy.io as scio
    scio.savemat(file, {'min': [i[0] for i in data], 'mean': [i[1] for i in data], 'max': [i[2] for i in data]})
    data = scio.loadmat(file)
    print(data)