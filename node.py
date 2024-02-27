# ======================================================================================================================
#   Class NODE
# ======================================================================================================================
class Node:

    def __init__(self):
        self.bus_i = -1                       # bus number (positive integer)
        self.type = -1                        # bus type:
                                              #    - PQ bus = 1,
                                              #    - PV bus = 2,
                                              #    - reference bus = 3,
                                              #    - isolated bus = 4
        self.gs = 0.0                         # Gs, shunt conductance (MW demanded at V = 1.0 p.u.)
        self.bs = 0.0                         # Bs, shunt susceptance (MVAr injected at V = 1.0 p.u.)
        self.base_kv = 0.0                    # baseKV, base voltage (kV)
        self.v_max = 1.10                     # maxVm, maximum voltage magnitude (p.u.)
        self.v_min = 0.90                     # minVm, minimum voltage magnitude (p.u.)
        self.flexibility = NodeFlexibility()  # Flexibility structure


# ======================================================================================================================
#   Class NODE FLEXIBILITY
# ======================================================================================================================
class NodeFlexibility:

    def __init__(self):
        self.upward = list()                # Note: FL - increase consumption
        self.downward = list()
        self.cost = list()
