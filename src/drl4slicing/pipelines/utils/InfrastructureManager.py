import networkx as ntx
import numpy as np
# from copy import deepcopy


class InfrastructureManager:
    def __init__(self,
                 expected_cnodes: int,
                 expected_switches: int,
                 expected_routers: int,
                 min_cpu: float,
                 max_cpu: float,
                 min_ram: float,
                 max_ram: float,
                 min_stor: float,
                 max_stor: float,
                 cnodePl_min_bw: float,
                 cnodePl_max_bw: float,
                 corePl_min_bw: float,
                 corePl_max_bw: float):
        #===================================
        self.expected_cnodes = expected_cnodes
        self.expected_switches = expected_switches
        self.expected_routers = expected_routers
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.min_ram = min_ram
        self.max_ram = max_ram
        self.min_stor = min_stor
        self.max_stor = max_stor
        self.cnodePl_min_bw = cnodePl_min_bw
        self.cnodePl_max_bw = cnodePl_max_bw
        self.corePl_min_bw = corePl_min_bw
        self.corePl_max_bw = corePl_max_bw
        #===================================
        self.current_nspr = None
        self.ongoing_vnf_id = -1
        self.infrastructure = None
        self.adjacency_matrix = None
        self.number_cnodes = None
        self.number_switches = None
        self.number_routers = None


    def reset(self, infrastructure_id, seed):
        if infrastructure_id == 1:
            self.infrastructure, self.number_cnodes, self.number_switches, self.number_routers = self.infrastructure1(seed)
        #===================================
        assert self.number_cnodes == self.expected_cnodes, \
        "Expected number of cnodes (set in parameters.yml) does not match number of cnodes found in infrastructure"
        assert self.number_switches == self.expected_switches, \
        "Expected number of switches (set in parameters.yml) does not match number of switches found in infrastructure"
        assert self.number_routers == self.expected_routers, \
        "Expected number of routers (set in parameters.yml) does not match number of routers found in infrastructure"
        #===================================
        self.current_nspr = None
        self.ongoing_vnf_id = None


    def load_nspr(self, nspr):
        self.current_nspr = nspr
        self.ongoing_vnf_id = 1
    

    def move_to_next_vnf(self):
        self.ongoing_vnf_id += 1
    

    def _node_bw(self, cnode_id: str):
        neigbors = list(self.infrastructure[cnode_id].keys())
        bandwidth = 0.0
        for neigbor in neigbors:
            bandwidth += self.infrastructure[cnode_id][neigbor].get("bw")
        #===================================
        return bandwidth
    

    def _is_previous_vnf_host(self, cnode_id: str):
        if self.ongoing_vnf_id == 1:
            return 0.0
        else: # i.e, self.ongoing_vnf_id >= 2
            if self.current_nspr.nodes[f"vnf{self.ongoing_vnf_id-1}"]["host"] == cnode_id:
                return 1.0
            else:
                return 0.0
    

    def _vnf_bw(self, vnf_id: int):
        if vnf_id == 1:
            return 0.0
        else:
            return self.current_nspr[f"vnf{vnf_id-1}"][f"vnf{vnf_id}"]["rq_bw"]
    

    def _is_first_vnf(self, vnf_id: int):
        if vnf_id == 1:
            return 1.0
        else:
            return 0.0


    def describe_only_cnodes_and_vnf_requirements(self):
        cnodes_resources = []
        for i in range(self.number_cnodes):
            cnode = self.infrastructure.nodes[f"cnode{i+1}"]
            cnodes_resources.append( [cnode["cpu"], cnode["ram"], cnode["stor"], self._node_bw(f"cnode{i+1}"), self._is_previous_vnf_host(f"cnode{i+1}")] )
        #===================================
        vnf = self.current_nspr.nodes[f"vnf{self.ongoing_vnf_id}"]
        vnf_requirements = [vnf["rq_cpu"], vnf["rq_ram"], vnf["rq_stor"], self._vnf_bw(self.ongoing_vnf_id), self._is_first_vnf(self.ongoing_vnf_id)]
        # print(cnodes_resources)
        # print(vnf_requirements)
        #===================================
        return [cnodes_resources, vnf_requirements]


    def describe(self):
        return self.describe_only_cnodes_and_vnf_requirements()
    

    def try_placement(self, suggested_cnode_id): # suggested_cnode_id = index (from 0) of cnode where to place current VNF
        suggested_cnode = self.infrastructure.nodes[f"cnode{suggested_cnode_id+1}"]
        vnf_to_place = self.current_nspr.nodes[f"vnf{self.ongoing_vnf_id}"]
        #===================================
        placed = False
        reward = -100.0
        if self.cnode_has_sufficient_resources(suggested_cnode, vnf_to_place):
            reward = 100.0 * ((suggested_cnode["cpu"]/self.max_cpu) + (suggested_cnode["ram"]/self.max_ram) + (suggested_cnode["stor"]/self.max_stor))
            if self.ongoing_vnf_id == 1:
                placed = True
                self.allocate_cnode(f"cnode{suggested_cnode_id+1}", vnf_to_place)
                self.current_nspr.nodes["vnf1"]["host"] = f"cnode{suggested_cnode_id+1}"
            else:
                path_to_previous_vnf = self.find_a_valid_path_between(self.current_nspr.nodes[f"vnf{self.ongoing_vnf_id-1}"]["host"], f"cnode{suggested_cnode_id+1}", self._vnf_bw(self.ongoing_vnf_id))
                if path_to_previous_vnf is not None:
                    placed = True
                    reward *= (1.0 / (len(path_to_previous_vnf) - 1))
                    self.allocate_cnode(f"cnode{suggested_cnode_id+1}", vnf_to_place)
                    self.current_nspr.nodes[f"vnf{self.ongoing_vnf_id}"]["host"] = f"cnode{suggested_cnode_id+1}"
                    self.allocate_path(path_to_previous_vnf, self._vnf_bw(self.ongoing_vnf_id))
        #===================================
        islastvnf = (self.ongoing_vnf_id == len(self.current_nspr))
        #===================================
        return placed, reward, islastvnf
    

    def find_a_valid_path_between(self, cnode_id1, cnode_id2, bandwidth):
        # mBFS : return a list of the nodes in the path if path found, otherwise return None
        if cnode_id1 == cnode_id2:
            return []
        #===================================
        computed_paths = []
        a_path = None
        neighbors = []
        node_name = cnode_id1
        computed_paths.append([cnode_id1])
        #===================================
        while len(computed_paths) != 0:
            a_path = computed_paths.pop(0)
            node_name = a_path[-1]
            if node_name == cnode_id2:
                return a_path
            for neighbor in list(self.infrastructure[node_name].keys()):
                if neighbor not in a_path and self.infrastructure[node_name][neighbor]["bw"] >= bandwidth:
                    neighbors.append(neighbor)
            for neighbor in neighbors:
                computed_paths.append(a_path+[neighbor])
            neighbors.clear()
        #===================================
        return None
    

    def allocate_path(self, a_path, bandwidth):
        for i in range(len(a_path)-1):
            self.infrastructure[a_path[i]][a_path[i+1]]["bw"] -= bandwidth
    

    def cnode_has_sufficient_resources(self, cnode, vnf):
        if cnode["cpu"] >= vnf["rq_cpu"] and \
            cnode["ram"] >= vnf["rq_ram"] and \
            cnode["stor"] >= vnf["rq_stor"]:
            return True
        else:
            return False
    

    def allocate_cnode(self, cnode_id: str, vnf):
        self.infrastructure.nodes[cnode_id]["cpu"] -= vnf["rq_cpu"]
        self.infrastructure.nodes[cnode_id]["ram"] -= vnf["rq_ram"]
        self.infrastructure.nodes[cnode_id]["stor"] -= vnf["rq_stor"]
    

    def infrastructure1(self, seed):
        np_gen = np.random.default_rng(seed)
        infrastructure = ntx.Graph()
        #===================================
        # create computing nodes
        number_cnodes = 6
        for i in range(1, number_cnodes + 1):
            cpu_value = np_gen.uniform(self.min_cpu, self.max_cpu)
            ram_value = np_gen.uniform(self.min_ram, self.max_ram)
            stor_value = np_gen.uniform(self.min_stor, self.max_stor)
            infrastructure.add_node("cnode"+str(i), cpu=cpu_value, ram=ram_value, stor=stor_value)
        #===================================
        # create switches
        number_switches = 3
        for i in range(1, number_switches + 1):
            infrastructure.add_node("switch"+str(i))
        #===================================
        # create routers
        number_routers = 4
        for i in range(1, number_routers + 1):
            infrastructure.add_node("router"+str(i))
        #===================================
        # link computing nodes to switches
        cnodes_ids          = [1, 2, 3, 4, 5, 6]
        switches_to_link_to = [1, 1, 2, 2, 3, 3]
        for cnode_id, switch_id in zip(cnodes_ids, switches_to_link_to):
            bw_value = np_gen.uniform(self.cnodePl_min_bw, self.cnodePl_max_bw)
            infrastructure.add_edge("cnode"+str(cnode_id), "switch"+str(switch_id), bw=bw_value)
        #===================================
        # link switches to their routers
        infrastructure.add_edge("switch1", "router1", bw=np_gen.uniform(self.corePl_min_bw, self.corePl_max_bw))
        infrastructure.add_edge("switch2", "router3", bw=np_gen.uniform(self.corePl_min_bw, self.corePl_max_bw))
        infrastructure.add_edge("switch3", "router4", bw=np_gen.uniform(self.corePl_min_bw, self.corePl_max_bw))
        #===================================
        # link routers between themselves
        routers_ids        = [1, 1, 1, 2, 2, 3]
        routers_to_link_to = [2, 3, 4, 3, 4, 4]
        for router_x, router_y in zip(routers_ids, routers_to_link_to):
            bw_value = np_gen.uniform(self.corePl_min_bw, self.corePl_max_bw)
            infrastructure.add_edge("router"+str(router_x), "router"+str(router_y), bw=bw_value)
        #===================================
        return infrastructure, number_cnodes, number_switches, number_routers