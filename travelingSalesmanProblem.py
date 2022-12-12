import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
## Algorithms
### Branch and Bound  
class Node:
    def __init__(self, bound, boundEdges, cost, solution):
        self.bound = bound
        self.boundEdges = boundEdges
        self.cost = cost
        self.solution = solution
    
    def __lt__(self, other):
        if len(self.solution) == len(other.solution):
            return self.bound < other.bound
        return len(self.solution) > len(other.solution)
    def __repr__(self) -> str:
        return f"Node({self.bound}, {self.boundEdges}, {self.cost}, {self.solution})"

def findTwoMinimalEdges(list):
    min1 = np.inf
    min2 = np.inf
    for j in list:
        if list[j]['weight'] < min1:
            min2 = min1
            min1 = list[j]['weight']
        elif list[j]['weight'] < min2:
            min2 = list[j]['weight']
    return min1, min2

def findInitialBound(A):
    bound = 0
    initialBoundEdges = np.zeros((A.number_of_nodes(), 2), dtype=list)
    for i in range(A.number_of_nodes()):
        min1, min2 = findTwoMinimalEdges(A[i])
        initialBoundEdges[i][0] = min1
        initialBoundEdges[i][1] = min2
        bound += min1 + min2
    return bound / 2, initialBoundEdges

def findBound(A, solution, boundEdges, bound):
    changedEdges = np.zeros(A.number_of_nodes(), dtype=int)
    newEdges = np.array(boundEdges)
    edgeWeight = A[solution[-2]][solution[-1]]['weight']
    sum = bound * 2
    if newEdges[solution[-2]][0] != edgeWeight:
        if changedEdges[solution[-2]] == 0:
            sum -= newEdges[solution[-2]][1]
            sum += edgeWeight
        else:
            sum -= newEdges[solution[-2]][0]
            sum += edgeWeight
        changedEdges[solution[-2]] += 1
    if newEdges[solution[-1]][0] != edgeWeight:
        if changedEdges[solution[-1]] == 0:
            sum -= newEdges[solution[-1]][1]
            sum += edgeWeight
        else:
            sum -= newEdges[solution[-1]][0]
            sum += edgeWeight
        changedEdges[solution[-1]] += 1
    return sum / 2, newEdges
from heapq import heappush, heappop

def branchAndBoundTSP(A):
    initialBound, initialBoundEdges = findInitialBound(A)
    root = Node(initialBound, initialBoundEdges, 0, [0])
    heap = []
    heappush(heap, root)
    best = np.inf
    solution = []
    nodeCount = 0
    while heap:
        node = heappop(heap)
        nodeCount += 1
        level = len(node.solution)
        if level > A.number_of_nodes():
            if best > node.cost:
                best = node.cost
                solution = node.solution
        else:
            if node.bound < best:
                if level < A.number_of_nodes() - 2:
                    for k in range(1, A.number_of_nodes()):
                        if k == node.solution[-1] or k == 0:
                            continue
                        edgeWeight = A[node.solution[-1]][k]['weight']
                        newBound, newEdges = findBound(A, node.solution + [k], node.boundEdges, node.bound) 
                        if k not in node.solution and newBound < best:
                            newNode = Node(newBound, newEdges, node.cost + edgeWeight, node.solution + [k])
                            if k == 2:
                                if 1 not in node.solution:  
                                    continue 
                            heappush(heap, newNode)
                else:
                    for k in range(1, A.number_of_nodes()):
                        if k == node.solution[-1] or k == 0:
                            continue
                        lastNode = 0
                        for i in range(1, A.number_of_nodes()):
                            if i not in node.solution + [k] and k != i:
                                lastNode = i
                                break
                        edgeWeight = A[node.solution[-1]][k]['weight']
                        nextEdgeWeight = A[k][lastNode]['weight']
                        lastEdgeWeight = A[lastNode][0]['weight']
                        cost = node.cost + edgeWeight + nextEdgeWeight + lastEdgeWeight
                        if k not in node.solution and cost < best:
                            newNode = Node(cost, [], cost, node.solution + [k, lastNode, 0])
                            heappush(heap, newNode)
    return best, solution
### Twice Around The Tree
def findPathWeight(A, path):
    weight = 0
    for i in range(len(path) - 1):
        weight += A[path[i]][path[i + 1]]['weight']
    return weight

def twiceAroundTheTreeTSP(A):
    MST = nx.minimum_spanning_tree(A)
    path = nx.dfs_preorder_nodes(MST, 0)
    path = list(path)
    path.append(path[0])
    weight = findPathWeight(A, path)
    return weight, path
### Christofides
def findShortcutPath(A):
    path = list(nx.eulerian_circuit(A, 0))
    path = [x[0] for x in path]

    # remove duplicates
    shortcutPath = list(dict.fromkeys(path))
    
    return shortcutPath + [shortcutPath[0]]

def christofidesTSP(A):
    MST = nx.minimum_spanning_tree(A)
    degrees = nx.degree(MST)
    oddNodes = [x[0] for x in degrees if degrees[x[0]] % 2 == 1]
    oddNodesSubgraph = nx.subgraph(A, oddNodes)
    matching = list(nx.min_weight_matching(oddNodesSubgraph, maxcardinality=True))

    MSTMultiGraph = nx.MultiGraph(MST)
    for i in range(len(matching)):
        node1 = matching[i][0]
        node2 = matching[i][1]
        MSTMultiGraph.add_edge(node1, node2, weight=A[node1][node2]['weight'])

    path = findShortcutPath(MSTMultiGraph)
    weight = findPathWeight(A, path)

    return weight, path
## Testing the algorithms
def findEuclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def findManhattanDistance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def generateInstances(n):
    euclideanGraph = nx.complete_graph(n)
    manhattanGraph = nx.complete_graph(n)
    nodes = []
    for i in range(n):
        newX = np.random.randint(1, n)
        newY = np.random.randint(1, n)
        for j in range(len(nodes)):
            if j == i:
                euclideanGraph[i][j]['weight'] = 0
                manhattanGraph[i][j]['weight'] = 0
            else:
                x = nodes[j][0]
                y = nodes[j][1]
                euclideanDistance = findEuclideanDistance(x, y, newX, newY)
                manhattanDistance = findManhattanDistance(x, y, newX, newY)
                if euclideanDistance == 0:
                    euclideanDistance = 1
                if manhattanDistance == 0:
                    manhattanDistance = 1
                euclideanGraph[i][j]['weight'] = euclideanDistance
                manhattanGraph[i][j]['weight'] = manhattanDistance
        nodes.append((newX, newY))
    return euclideanGraph, manhattanGraph

def drawGraph(A):
    layout = nx.spring_layout(A)
    nx.draw(A, layout)
    nx.draw_networkx_edge_labels(A, pos=layout, edge_labels=nx.get_edge_attributes(A, 'weight'))
    labels = {i: i for i in range(len(A))}
    nx.draw_networkx_labels(A, pos=layout, labels=labels)
    plt.show()

def printSolution(A, weight, solution, algorithm):
    print(algorithm)
    print('Path: ', end='')
    for i in range(len(solution)):
        print(solution[i], end='')
        if i != len(solution) - 1:
            print(' -> ', end='')
    print()
    print('Weight: ', weight)

def printSolutionsReport(A):
    print('_' * 50)
    print()
    print('Number of nodes: ', A.number_of_nodes())
    print('Number of edges: ', A.number_of_edges())
    print()
    weight, solution = branchAndBoundTSP(A)
    printSolution(A, weight, solution, 'Branch and bound')
    print()
    weight, solution = twiceAroundTheTreeTSP(A)
    printSolution(A, weight, solution, 'Twice Around The Tree')
    print()
    weight, solution = christofidesTSP(A)
    printSolution(A, weight, solution, 'Christofides')
    print()
    print('_' * 50)
### Test 1
A = [[0, 3, 1, 5, 8],
    [3, 0, 6, 7, 9],
    [1, 6, 0, 4, 2],
    [5, 7, 4, 0, 3],
    [8, 9, 2, 3, 0]]

A = nx.from_numpy_matrix(np.matrix(A), create_using=nx.Graph)

drawGraph(A)

printSolutionsReport(A)
### Test 2
G = [[0, 4, 8, 9, 12],
    [4, 0, 6, 8, 9],
    [8, 6, 0, 10, 11],
    [9, 8, 10, 0, 7],   
    [12, 9, 11, 7, 0]]

G = nx.from_numpy_matrix(np.matrix(G), create_using=nx.Graph)

drawGraph(G)

printSolutionsReport(G)
### Random Test
euclideanGraph, manhattanGraph = generateInstances(2**3)

drawGraph(euclideanGraph)

printSolutionsReport(euclideanGraph)
drawGraph(manhattanGraph)

printSolutionsReport(manhattanGraph)
## Experiments
def timeout(func, graph, timeout_duration=1):
    import signal
    class TimeoutError(Exception):
        pass
    def handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(graph)
    except TimeoutError as exc:
        result = None, None
    finally:
        signal.alarm(0)
    return result

def testAlgorithm(algorithm, A, timeoutDuration, printResults):
    weight, solution = timeout(algorithm, A, timeout_duration=timeoutDuration)
    if weight is None:
        print('Timeout for ', algorithm.__name__, ' algorithm after ', timeoutDuration, ' seconds.')
    else:
        if printResults:
            printSolution(A, weight, solution, algorithm.__name__)
import time
def testBranchAndBoundTime():
    meanEuclidean = 0
    meanManhattan = 0
    for i in range(10):
        euclideanGraph, manhattanGraph = generateInstances(2**4)
        
        start = time.time()
        testAlgorithm(branchAndBoundTSP, euclideanGraph, 30*60, False)
        end = time.time()
        meanEuclidean += end - start
        
        start = time.time()
        testAlgorithm(branchAndBoundTSP, manhattanGraph, 30*60, False)
        end = time.time()
        meanManhattan += end - start
    meanEuclidean /= 10
    meanManhattan /= 10
    print('Mean time for Branch and Bound on Euclidean graph: ', meanEuclidean)
    print('Mean time for Branch and Bound on Manhattan graph: ', meanManhattan)
    return meanEuclidean, meanManhattan

def testApproximativeTimes(algorithm):
    meanEuclidean = [0 for i in range(7)]
    meanManhattan = [0 for i in range(7)]
    for i in range(4, 11):
        for j in range(10):
            euclideanGraph, manhattanGraph = generateInstances(2**i)
            
            start = time.time()
            testAlgorithm(algorithm, euclideanGraph, 30*60, False)
            end = time.time()
            meanEuclidean[i - 4] += end - start
            
            start = time.time()
            testAlgorithm(algorithm, manhattanGraph, 30*60, False)
            end = time.time()
            meanManhattan[i - 4] += end - start
    for i in range(7):
        meanEuclidean[i] /= 10
        meanManhattan[i] /= 10
    print(f'Mean time for {algorithm.__name__} on Euclidean graph: ', meanEuclidean)
    print(f'Mean time for {algorithm.__name__} on Manhattan graph: ', meanManhattan)
    return meanEuclidean, meanManhattan
def testApproximativeQuality(algorithm):
    meanEuclideanQuality = 0
    meanManhattanQuality = 0
    for i in range(1000):
        euclideanGraph, manhattanGraph = generateInstances(10)
        
        exactWeight, _ = branchAndBoundTSP(euclideanGraph)
        euclideanWeight, _ = algorithm(euclideanGraph)
        meanEuclideanQuality += euclideanWeight / exactWeight

        exactWeight, _ = branchAndBoundTSP(manhattanGraph)
        manhattanWeight, _ = algorithm(manhattanGraph)
        meanManhattanQuality += manhattanWeight / exactWeight

    meanEuclideanQuality /= 1000
    meanManhattanQuality /= 1000
    print(f'Mean quality for {algorithm.__name__} on Euclidean graph: ', meanEuclideanQuality)
    print(f'Mean quality for {algorithm.__name__} on Manhattan graph: ', meanManhattanQuality)
    return meanEuclideanQuality, meanManhattanQuality

import matplotlib.pyplot as plt
def plotTimes(euclideanTimes, manhattanTimes, algorithm):
    plt.plot([2**i for i in range(4, 11)], euclideanTimes, label=f'{algorithm.__name__} Euclidiano')
    plt.plot([2**i for i in range(4, 11)], manhattanTimes, label=f'{algorithm.__name__} Manhattan')
    plt.xlabel('Número de vértices')
    plt.ylabel('Tempo (s)')
    plt.title(f'Tempo de execução do algoritmo {algorithm.__name__}')
    plt.legend()
    plt.show()
### Testing the time performance of the algorithms
testBranchAndBoundTime()
euclideanTimes, manhattanTimes = testApproximativeTimes(twiceAroundTheTreeTSP)
plotTimes(euclideanTimes, manhattanTimes, twiceAroundTheTreeTSP)
euclideanTimes, manhattanTimes = testApproximativeTimes(christofidesTSP)
plotTimes(euclideanTimes, manhattanTimes, christofidesTSP)
### Testing how distant the approximations are from the optimal solution
testApproximativeQuality(twiceAroundTheTreeTSP)
testApproximativeQuality(christofidesTSP)