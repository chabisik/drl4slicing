import networkx as ntx
import numpy as np


class NSPRGenerator:
    def __init__(self,
                 rq_min_cpu: float,
                 rq_max_cpu: float,
                 rq_min_ram: float,
                 rq_max_ram: float,
                 rq_min_stor: float,
                 rq_max_stor: float,
                 rq_min_bw: float,
                 rq_max_bw: float,
                 min_vnfs: int,
                 max_vnfs: int,
                 min_batch_nsprs: int,
                 max_batch_nsprs: int):
        #===================================
        self.rq_min_cpu = rq_min_cpu
        self.rq_max_cpu = rq_max_cpu
        self.rq_min_ram = rq_min_ram
        self.rq_max_ram = rq_max_ram
        self.rq_min_stor = rq_min_stor
        self.rq_max_stor = rq_max_stor
        self.rq_min_bw = rq_min_bw
        self.rq_max_bw = rq_max_bw
        self.min_vnfs = min_vnfs
        self.max_vnfs = max_vnfs
        self.min_batch_nsprs = min_batch_nsprs
        self.max_batch_nsprs = max_batch_nsprs
        #===================================
        self.np_gen = None
    

    def reset(self, seed):
        self.np_gen = np.random.default_rng(seed)
    

    def get_batch_nsprs(self):
        n_nsprs = self.np_gen.integers(self.min_batch_nsprs, self.max_batch_nsprs + 1)
        assert n_nsprs >= 1, "at least one nspr should be generated"
        nsprs = []
        #===================================
        for _ in range(n_nsprs):
            nspr = ntx.DiGraph()
            n_vnfs = self.np_gen.integers(self.min_vnfs, self.max_vnfs + 1)
            assert n_vnfs >= 2, "a network slice placement request should contain at least two vnfs"
            for i in range(1, n_vnfs + 1):
                rq_cpu = self.np_gen.uniform(self.rq_min_cpu, self.rq_max_cpu)
                rq_ram = self.np_gen.uniform(self.rq_min_ram, self.rq_max_ram)
                rq_stor = self.np_gen.uniform(self.rq_min_stor, self.rq_max_stor)
                nspr.add_node(f"vnf{i}", rq_cpu=rq_cpu, rq_ram=rq_ram, rq_stor=rq_stor)
                if i > 1:
                    rq_bw = self.np_gen.uniform(self.rq_min_bw, self.rq_max_bw)
                    nspr.add_edge(f"vnf{i-1}", f"vnf{i}", rq_bw=rq_bw)
                
            nsprs.append(nspr)
        # a batch of network slice placement requests
        #===================================
        return nsprs