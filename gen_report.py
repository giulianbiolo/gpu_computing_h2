''' Generate matplotlib graphs for the report with the given data
-----------------------------------------------------------------
Input Data: Written text in the file 'transpose-0.txt'
Output Data: Graphs in the folder 'graphs'
-----------------------------------------------------------------
Input Format:
Naive Transpose:
N, OpTime[ms], OpThroughput[GB/s], KTime[ms], KThroughput[GB/s]
int, float, float, float, float\n
int, float, float, float, float\n
...
\n\n
Block Transpose with conflicts:
N, OpTime[ms], OpThroughput[GB/s], KTime[ms], KThroughput[GB/s]
int, float, float, float, float\n
int, float, float, float, float\n
...
\n\n
Block Transpose without coalesced:
N, OpTime[ms], OpThroughput[GB/s], KTime[ms], KThroughput[GB/s]
int, float, float, float, float\n
int, float, float, float, float\n
...
\n\n
EOF
-----------------------------------------------------------------
Output Format:
> Graph with N on x-axis and KThroughput on y-axis ( With 3 different colored lines, one for each method, with legend )      # shows throughput of the kernel only
> Graph with N on x-axis and OpThroughput on y-axis ( With 3 different colored lines, one for each method, with legend )     # shows throughput of the operations
> Graph with N on x-axis and (OpTime - KTime) on y-axis ( With 3 different colored lines, one for each method, with legend ) # shows time taken only by memory operations
> Graph with N on x-axis and (N*N*4*2) / (OpTime - KTime) on y-axis ( With 3 different colored lines, one for each method, with legend ) # shows memory throughput
Save these graphs in the folder 'graphs' under the names 'kthroughput.png', 'opthroughput.png', 'memtime.png'
'''

import matplotlib.pyplot as plt
import sys
import os


class GraphData:
    def __init__(self, data: list[str]):
        self.N = [int(i.split(',')[0]) for i in data]
        self.OpTime = [float(i.split(',')[1]) for i in data]
        self.OpThroughput = [float(i.split(',')[2]) for i in data]
        self.KTime = [float(i.split(',')[3]) for i in data]
        self.KThroughput = [float(i.split(',')[4]) for i in data]
        if (len(self.N) != len(self.OpTime) != len(self.OpThroughput) != len(self.KTime) != len(self.KThroughput)):
            raise ValueError("Data not in correct format")
    
    def __str__(self):
        return f'\n\tN: {self.N}\n\tOpTime: {self.OpTime}\n\tOpThroughput: {self.OpThroughput}\n\tKTime: {self.KTime}\n\tKThroughput: {self.KThroughput}\n'
    def __repr__(self):
        return f'\n\tN: {self.N}\n\tOpTime: {self.OpTime}\n\tOpThroughput: {self.OpThroughput}\n\tKTime: {self.KTime}\n\tKThroughput: {self.KThroughput}\n'


def load_data(filename: str) -> list[GraphData]:
    graphs_data: list[GraphData] = []
    with open(filename, 'r', encoding = "UTF-8") as file:
        raw_blocks: list[str] = file.read().split('\n\n')
        raw_blocks = [x for x in raw_blocks if x.strip() != '']
        for block in raw_blocks:
            to_process: list[str] = []
            for line in [b.strip() for b in block.split('\n') if b.strip() != '']:
                if "N" in line.split("\n")[0] or "B" in line.split("\n")[0]: continue
                to_process.append(line)
            print("Now processing: ", to_process)
            graphs_data.append(GraphData(to_process))
    return graphs_data

def plot_kthroughput(data: list[GraphData]) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(19.20, 10.80)
    labels = ['Naive Transpose', 'Block Transpose With Bank Conflicts', 'Block Transpose Coalesced']
    for i in range(len(data)):
        ax.plot([i for i in range(0, len(data[i].N))], data[i].KThroughput, label = labels[i])
    ax.set_title('Kernel Only Throughput over Matrix Size')
    ax.set_xlabel('Matrix Size NxN')
    ax.set_ylabel('Kernel Only Throughput [ GB/s ]')
    ax.set_xticks([i for i in range(0, len(data[i].N))])
    ax.set_xticklabels(data[i].N)
    ax.legend()
    plt.savefig(f'graphs/kthroughput.png')
    plt.show()

def plot_opthroughput(data: list[GraphData]) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(19.20, 10.80)
    labels = ['Naive Transpose', 'Block Transpose With Bank Conflicts', 'Block Transpose Coalesced']
    for i in range(len(data)):
        ax.plot([i for i in range(0, len(data[i].N))], data[i].OpThroughput, label = labels[i])
    ax.set_title('Operation Throughput over Matrix Size')
    ax.set_xlabel('Matrix Size NxN')
    ax.set_ylabel('Operation Throughput [ GB/s ]')
    ax.set_xticks([i for i in range(0, len(data[i].N))])
    ax.set_xticklabels(data[i].N)
    ax.legend()
    plt.savefig(f'graphs/opthroughput.png')
    plt.show()

def plot_memtime(data: list[GraphData]) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(19.20, 10.80)
    labels = ['Naive Transpose', 'Block Transpose With Bank Conflicts', 'Block Transpose Coalesced']
    for i in range(len(data)):
        ax.plot(data[i].N, [data[i].OpTime[j] - data[i].KTime[j] for j in range(len(data[i].N))], label = labels[i])
    ax.set_title('Memory Time over Matrix Size')
    ax.set_xlabel('Matrix Size NxN')
    ax.set_ylabel('Memory Time [ ms ]')
    ax.legend()
    plt.savefig(f'graphs/memtime.png')
    plt.show()

def plot_memthroughput(data: list[GraphData]) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(19.20, 10.80)
    labels = ['Naive Transpose', 'Block Transpose With Bank Conflicts', 'Block Transpose Coalesced']
    for i in range(len(data)):
        # optime - ktime = memtime, we want memthroughput which is N*N*4*2 / memtime
        mem_throughput = [(data[i].N[j] * data[i].N[j] * 4 * 2) / (data[i].OpTime[j] - data[i].KTime[j]) for j in range(len(data[i].N))]
        ax.plot(data[i].N, mem_throughput, label = labels[i])
    ax.set_title('Memory Throughput over Matrix Size')
    ax.set_xlabel('Matrix Size NxN')
    ax.set_ylabel('Memory Throughput [ GB/s ]')
    ax.legend()
    plt.savefig(f'graphs/memthroughput.png')
    plt.show()
    # also plot it only to N[:5]
    fig, ax = plt.subplots()
    fig.set_size_inches(19.20, 10.80)
    labels = ['Naive Transpose', 'Block Transpose With Bank Conflicts', 'Block Transpose Coalesced']
    for i in range(len(data)):
        # optime - ktime = memtime, we want memthroughput which is N*N*4*2 / memtime
        mem_throughput = [(data[i].N[j] * data[i].N[j] * 4 * 2) / (data[i].OpTime[j] - data[i].KTime[j]) for j in range(len(data[i].N))]
        ax.plot(data[i].N[:5], mem_throughput[:5], label = labels[i])
    ax.set_title('Memory Throughput over Matrix Size (Only up to 1024)')
    ax.set_xlabel('Matrix Size NxN')
    ax.set_ylabel('Memory Throughput [ GB/s ]')
    ax.legend()
    plt.savefig(f'graphs/memthroughput_upto1024.png')
    plt.show()

def main(filename: str) -> None:
    data: list[GraphData] = load_data(filename)
    print(data)
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    plot_kthroughput(data)
    plot_opthroughput(data)
    plot_memtime(data)
    plot_memthroughput(data)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 gen_report.py <filename>")
        sys.exit(1)
    filename: str = sys.argv[1]
    main(filename)
