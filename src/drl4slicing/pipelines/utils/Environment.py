class Environment:
    def __init__(self, infrastructure_manager, nsprs_generator):
        self.infrastructure_manager = infrastructure_manager
        self.nsprs_generator = nsprs_generator
        self.waiting_nsprs = []
        #===================================
        self.n_successful_nsprs = 0
    

    def reset(self, infrastructure_id, infrastructure_seed, nspr_seed):
        self.infrastructure_manager.reset(infrastructure_id, infrastructure_seed)
        self.nsprs_generator.reset(nspr_seed)

        self.waiting_nsprs.clear()
        self.waiting_nsprs.extend( self.nsprs_generator.get_batch_nsprs()[:] )

        self.infrastructure_manager.load_nspr( self.waiting_nsprs.pop(0) )

        self.n_successful_nsprs = 0

        return self.infrastructure_manager.describe()


    def step(self, action):
        placed, reward, islastvnf = self.infrastructure_manager.try_placement(action)

        if placed:
            if islastvnf:
                self.n_successful_nsprs += 1
                if len(self.waiting_nsprs) == 0:
                    self.waiting_nsprs.extend( self.nsprs_generator.get_batch_nsprs()[:] )
                self.infrastructure_manager.load_nspr( self.waiting_nsprs.pop(0) )
            else:
                self.infrastructure_manager.move_to_next_vnf()

        return self.infrastructure_manager.describe(), reward, (not placed), "any info"


    def statistics(self):
        return self.n_successful_nsprs