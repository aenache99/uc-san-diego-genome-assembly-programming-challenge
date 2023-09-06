# python 3
from collections import deque
from typing import List, Optional


class EdgeFG:
    def __init__(self, start, end, capacity):
        self.start = start
        self.end = end
        self.capacity = capacity
        self.flow = 0

    def __str__(self):
        s = "Edge(start={}, end={}, capacity={}, flow={})" \
            .format(self.start + 1, self.end + 1, self.capacity, self.flow)
        return s

    def __repr__(self):
        s = "\nEdge(start={}, end={}, capacity={}, flow={})" \
            .format(self.start + 1, self.end + 1, self.capacity, self.flow)
        return s


class FlowGraph:
    INF = 10 ** 6

    def __init__(self, n):
        self.n = n
        self.edges = []
        self.graph = [[] for _ in range(n)]

    def add_edge(self, start, end, capacity):
        forward_edge = EdgeFG(start, end, capacity)
        backward_edge = EdgeFG(end, start, 0)
        self.graph[start].append(len(self.edges))
        self.edges.append(forward_edge)
        self.graph[end].append(len(self.edges))
        self.edges.append(backward_edge)

    def find_shortest_path(self, start, end) -> Optional[List[int]]:
        prev_edge = [None] * self.n
        processed = [False] * self.n
        processed[start] = True
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == end:
                break
            for edge_id in self.graph[cur]:
                edge = self.edges[edge_id]
                if (not processed[edge.end]) and (edge.flow < edge.capacity):
                    q.append(edge.end)
                    prev_edge[edge.end] = edge_id
                    processed[edge.end] = True
        path = None
        if prev_edge[end] is not None:
            path = []
            node = end
            while node != start:
                edge_id = prev_edge[node]
                path.append(edge_id)
                node = self.edges[edge_id].start
            path = path[::-1]
        return path

    def maximize_flow(self, start, end):
        while True:
            path = self.find_shortest_path(start, end)
            if path is None:
                break
            min_flow = FlowGraph.INF
            for edge_id in path:
                edge = self.edges[edge_id]
                min_flow = min(min_flow, edge.capacity - edge.flow)
            for edge_id in path:
                if edge_id % 2 == 0:
                    self.edges[edge_id].capacity -= min_flow
                    self.edges[edge_id + 1].capacity += min_flow
                else:
                    self.edges[edge_id - 1].capacity += min_flow
                    self.edges[edge_id].capacity -= min_flow
        for i in range(len(self.edges) // 2):
            self.edges[i * 2].flow = self.edges[i * 2 + 1].capacity
            self.edges[i * 2].capacity += self.edges[i * 2 + 1].capacity


class EdgeNC:
    def __init__(self, start, end, low_bound, capacity):
        self.start = start
        self.end = end
        self.low_bound = low_bound
        self.capacity = capacity
        self.flow = 0

    def __str__(self):
        s = "Edge(start={}, end={}, low_bound={}, capacity={})" \
            .format(self.start, self.end, self.low_bound, self.capacity)
        return s

    def __repr__(self):
        s = "Edge(start={}, end={}, low_bound={}, capacity={})" \
            .format(self.start, self.end, self.low_bound, self.capacity)
        return s


class NetworkCirculation:
    def __init__(self, n_vertices, edges):
        self.n = n_vertices
        self.edges = edges
        self.m = len(self.edges)

    def solve(self) -> Optional[List[int]]:
        if self.check_naive_solution():
            circulation = []
            for i, edge in enumerate(self.edges):
                edge.flow = edge.low_bound
                circulation.append(edge.flow)
            return circulation

        adj_list_prev = [[] for _ in range(self.n)]
        adj_list_next = [[] for _ in range(self.n)]
        for i, edge in enumerate(self.edges):
            adj_list_prev[edge.end].append(i)
            adj_list_next[edge.start].append(i)

        fg = FlowGraph(self.n + 2)
        for edge in self.edges:
            fg.add_edge(edge.start, edge.end, edge.capacity - edge.low_bound)

        demands = [0] * self.n
        for v in range(self.n):
            demands[v] -= sum([self.edges[i].low_bound for i in adj_list_prev[v]])
            demands[v] += sum([self.edges[i].low_bound for i in adj_list_next[v]])

        for v in range(self.n):
            demand = demands[v]
            if demand < 0:
                fg.add_edge(self.n, v, abs(demand))
            elif demand > 0:
                fg.add_edge(v, self.n + 1, demand)

        fg.maximize_flow(start=self.n, end=self.n + 1)

        for edge in fg.edges:
            if (edge.start == self.n) and (edge.flow != edge.capacity):
                return None
            elif (edge.end == self.n + 1) and (edge.flow != edge.capacity):
                return None

        circulation = []
        for i, edge in enumerate(self.edges):
            edge.flow = edge.low_bound + fg.edges[i * 2].flow
            circulation.append(edge.flow)
        return circulation

    def check_naive_solution(self) -> bool:
        old_flow = []
        for edge in self.edges:
            old_flow.append(edge.flow)
            edge.flow = edge.low_bound
        ans = self.check_solution(self.n, self.edges)
        for i, edge in enumerate(self.edges):
            edge.flow = old_flow[i]
        return ans

    @staticmethod
    def check_solution(n: int, edges: List[EdgeNC]) -> bool:
        for edge in edges:
            if edge.low_bound <= edge.flow <= edge.capacity:
                continue
            else:
                return False

        adj_list_prev = [[] for _ in range(n)]
        adj_list_next = [[] for _ in range(n)]
        for i, edge in enumerate(edges):
            adj_list_prev[edge.end].append(i)
            adj_list_next[edge.start].append(i)

        for v in range(n):
            inflow = sum([edges[i].flow for i in adj_list_prev[v]])
            outflow = sum([edges[i].flow for i in adj_list_next[v]])
            if inflow != outflow:
                return False
        return True


if __name__ == "__main__":
    n_vertices, n_edges = map(int, input().split())
    edges = []
    for _ in range(n_edges):
        v1, v2, low, cap = map(int, input().split())
        edges.append(EdgeNC(v1 - 1, v2 - 1, low, cap))
    edges = tuple(edges)

    circulation = NetworkCirculation(n_vertices, edges).solve()

    if circulation is not None:
        print("YES")
        for flow in circulation:
            print(flow)
    else:
        print("NO")
