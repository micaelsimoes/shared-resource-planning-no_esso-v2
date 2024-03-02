import os
from math import acos, tan, sqrt, atan2, isclose, pi
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pyomo.opt as po
import pyomo.environ as pe
from node import Node
from branch import Branch
from generator import Generator
from energy_storage import EnergyStorage
from helper_functions import *


# ======================================================================================================================
#   Class NETWORK
# ======================================================================================================================
class Network:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.diagrams_dir = str()
        self.year = int()
        self.num_instants = 0
        self.operational_data_file = str()
        self.data_loaded = False
        self.is_transmission = False
        self.baseMVA = 100.0
        self.nodes = list()
        self.branches = list()
        self.generators = list()
        self.energy_storages = list()
        self.shared_energy_storages = list()
        self.prob_market_scenarios = list()             # Probability of market (price) scenarios
        self.prob_operation_scenarios = list()          # Probability of operation (generation and consumption) scenarios
        self.cost_energy_p = list()

    def build_model(self, params):
        _pre_process_network(self)
        return _build_model(self, params)

    def run_smopf(self, model, params, from_warm_start=False):
        return _run_smopf(self, model, params, from_warm_start=from_warm_start)

    def read_network_from_json_file(self):
        filename = os.path.join(self.data_dir, self.name, f'{self.name}_{self.year}.json')
        _read_network_from_json_file(self, filename)
        self.perform_network_check()

    def read_network_operational_data_from_file(self):
        filename = os.path.join(self.data_dir, self.name, self.operational_data_file)
        data = _read_network_operational_data_from_file(self, filename)
        _update_network_with_excel_data(self, data)

    def process_results(self, model, params, results=dict()):
        return _process_results(self, model, params, results=results)

    def process_results_interface_power_flow(self, model):
        return _process_results_interface_power_flow(self, model)

    def get_reference_node_id(self):
        for node in self.nodes:
            if node.type == BUS_REF:
                return node.bus_i
        print(f'[ERROR] Network {self.name}. No REF NODE found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_node_idx(self, node_id):
        for i in range(len(self.nodes)):
            if self.nodes[i].bus_i == node_id:
                return i
        print(f'[ERROR] Network {self.name}. Bus ID {node_id} not found! Check network model.')
        exit(ERROR_NETWORK_FILE)

    def get_node_type(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return node.type
        print(f'[ERROR] Network {self.name}. Node {node_id} not found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_node_voltage_limits(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return node.v_min, node.v_max
        print(f'[ERROR] Network {self.name}. Node {node_id} not found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def node_exists(self, node_id):
        for i in range(len(self.nodes)):
            if self.nodes[i].bus_i == node_id:
                return True
        return False

    def get_reference_gen_idx(self):
        ref_node_id = self.get_reference_node_id()
        for i in range(len(self.generators)):
            gen = self.generators[i]
            if gen.bus == ref_node_id:
                return i
        print(f'[ERROR] Network {self.name}. No REF NODE found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_gen_idx(self, node_id):
        for g in range(len(self.generators)):
            gen = self.generators[g]
            if gen.bus == node_id:
                return g
        print(f'[ERROR] Network {self.name}. No Generator in bus {node_id} found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_gen_type(self, gen_id):
        description = 'Unkown'
        for gen in self.generators:
            if gen.gen_id == gen_id:
                if gen.gen_type == GEN_REFERENCE:
                    description = 'Reference (TN)'
                elif gen.gen_type == GEN_CONV:
                    description = 'Conventional'
                elif gen.gen_type == GEN_RES_CONTROLLABLE:
                    description = 'RES (Generic, Controllable)'
                elif gen.gen_type == GEN_RES_SOLAR:
                    description = 'RES (Solar)'
                elif gen.gen_type == GEN_RES_WIND:
                    description = 'RES (Wind)'
                elif gen.gen_type == GEN_RES_OTHER:
                    description = 'RES (Generic, Non-controllable)'
                elif gen.gen_type == GEN_INTERCONNECTION:
                    description = 'Interconnection'
        return description

    def get_num_renewable_gens(self):
        num_renewable_gens = 0
        for generator in self.generators:
            if generator.gen_type in GEN_CURTAILLABLE_TYPES:
                num_renewable_gens += 1
        return num_renewable_gens

    def get_branch_idx(self, branch):
        for b in range(len(self.branches)):
            if self.branches[b].branch_id == branch.branch_id:
                return b
        print(f'[ERROR] Network {self.name}. No Branch connecting bus {branch.fbus} and bus {branch.tbus} found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def has_energy_storage_device(self, node_id):
        for energy_storage in self.energy_storages:
            if energy_storage.bus == node_id:
                return True
        return False

    def get_shared_energy_storage_idx(self, node_id):
        for i in range(len(self.shared_energy_storages)):
            shared_energy_storage = self.shared_energy_storages[i]
            if shared_energy_storage.bus == node_id:
                return i
        print(f'[ERROR] Network {self.name}. Node {node_id} does not have a shared energy storage system! Check network.')
        exit(ERROR_NETWORK_FILE)

    def compute_objective_function_value(self, model, params):
        return _compute_objective_function_value(self, model, params)

    def perform_network_check(self):
        _perform_network_check(self)

    def compute_series_admittance(self):
        for branch in self.branches:
            branch.g = branch.r / (branch.r ** 2 + branch.x ** 2)
            branch.b = -branch.x / (branch.r ** 2 + branch.x ** 2)

    def plot_diagram(self):
        _plot_networkx_diagram(self)


# ======================================================================================================================
#   NETWORK optimization functions
# ======================================================================================================================
def _build_model(network, params):

    s_base = network.baseMVA
    network.compute_series_admittance()
    ref_node_id = network.get_reference_node_id()

    model = pe.ConcreteModel()
    model.name = network.name

    # ------------------------------------------------------------------------------------------------------------------
    # Sets
    model.periods = range(network.num_instants)
    model.scenarios_market = range(len(network.prob_market_scenarios))
    model.scenarios_operation = range(len(network.prob_operation_scenarios))
    model.nodes = range(len(network.nodes))
    model.generators = range(len(network.generators))
    model.branches = range(len(network.branches))
    model.energy_storages = range(len(network.energy_storages))
    model.shared_energy_storages = range(len(network.shared_energy_storages))

    # ------------------------------------------------------------------------------------------------------------------
    # Decision variables
    # - Voltage
    model.e = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=1.0)
    model.f = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.e_actual = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=1.0)
    model.f_actual = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    #model.vmag_sqr = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=1.0)
    if params.slack_voltage_limits:
        model.slack_e_up = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_e_down = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_f_up = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_f_down = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    if params.relaxed_model:
        model.penalty_gen_vg = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    for i in model.nodes:
        node = network.nodes[i]
        e_lb, e_ub = -node.v_max, node.v_max
        f_lb, f_ub = -node.v_max, node.v_max
        if node.type == BUS_REF:
            if network.is_transmission:
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            model.e[i, s_m, s_o, p].setlb(e_lb)
                            model.e[i, s_m, s_o, p].setub(e_ub)
                            model.f[i, s_m, s_o, p].fix(0.0)
                            if params.slack_voltage_limits:
                                model.slack_f_up[i, s_m, s_o, p].fix(0.0)
                                model.slack_f_down[i, s_m, s_o, p].fix(0.0)
            else:
                ref_gen_idx = network.get_gen_idx(node.bus_i)
                vg = network.generators[ref_gen_idx].vg
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            model.e_actual[i, s_m, s_o, p].fix(vg)
                            model.f_actual[i, s_m, s_o, p].fix(0.0)
        else:
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.e[i, s_m, s_o, p].setlb(e_lb)
                        model.e[i, s_m, s_o, p].setub(e_ub)
                        model.f[i, s_m, s_o, p].setlb(f_lb)
                        model.f[i, s_m, s_o, p].setub(f_ub)

    # - Generation
    model.pg = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.qg = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    for g in model.generators:

        gen = network.generators[g]

        if gen.is_controllable():

            pg_ub, pg_lb = gen.pmax, gen.pmin
            qg_ub, qg_lb = gen.qmax, gen.qmin

            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        if gen.status[p] == 1:
                            model.pg[g, s_m, s_o, p] = (pg_lb + pg_ub) * 0.50
                            model.qg[g, s_m, s_o, p] = (qg_lb + qg_ub) * 0.50
                            model.pg[g, s_m, s_o, p].setlb(pg_lb)
                            model.pg[g, s_m, s_o, p].setub(pg_ub)
                            model.qg[g, s_m, s_o, p].setlb(qg_lb)
                            model.qg[g, s_m, s_o, p].setub(qg_ub)
                        else:
                            model.pg[g, s_m, s_o, p].fix(0.0)
                            model.qg[g, s_m, s_o, p].fix(0.0)
        else:
            # Non-conventional generator
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        init_pg = 0.0
                        init_qg = 0.0
                        if gen.status[p] == 1:
                            init_pg = gen.pg[s_o][p]
                            init_qg = gen.qg[s_o][p]
                        model.pg[g, s_m, s_o, p].fix(init_pg)
                        model.qg[g, s_m, s_o, p].fix(init_qg)
    if params.rg_curt:
        model.pg_curt = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        for g in model.generators:
            gen = network.generators[g]
            if gen.is_controllable():
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            model.pg_curt[g, s_m, s_o, p].fix(0.0)
            else:
                if gen.is_curtaillable():
                    # - Renewable Generation
                    for s_m in model.scenarios_market:
                        for s_o in model.scenarios_operation:
                            for p in model.periods:
                                init_pg = 0.0
                                if gen.status[p] == 1:
                                    init_pg = max(gen.pg[s_o][p], 0.0)
                                model.pg_curt[g, s_m, s_o, p].setub(init_pg)
                else:
                    # - Generator is not curtaillable (conventional RES, ref gen, etc.)
                    for s_m in model.scenarios_market:
                        for s_o in model.scenarios_operation:
                            for p in model.periods:
                                model.pg_curt[g, s_m, s_o, p].fix(0.0)

    # - Branch current (squared)
    model.iij_sqr = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    if params.slack_line_limits:
        model.slack_iij_sqr = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    for b in model.branches:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    if not network.branches[b].status == 1:
                        model.iij_sqr[b, s_m, s_o, p].fix(0.0)
                        if params.slack_line_limits:
                            model.slack_iij_sqr[b, s_m, s_o, p].fix(0.0)

    # - Loads
    model.pc = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals)
    model.qc = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals)
    for i in model.nodes:
        node = network.nodes[i]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    model.pc[i, s_m, s_o, p].fix(node.pd[s_o][p])
                    model.qc[i, s_m, s_o, p].fix(node.qd[s_o][p])
    if params.fl_reg:
        model.flex_p_up = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.flex_p_down = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        if params.fl_relax:
            model.penalty_flex_day_balance = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.00)
        for i in model.nodes:
            node = network.nodes[i]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        flex_up = node.flexibility.upward[p]
                        flex_down = node.flexibility.downward[p]
                        model.flex_p_up[i, s_m, s_o, p].setub(abs(flex_up))
                        model.flex_p_down[i, s_m, s_o, p].setub(abs(flex_down))
    if params.l_curt:
        model.pc_curt = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.qc_curt = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        for i in model.nodes:
            node = network.nodes[i]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.pc_curt[i, s_m, s_o, p].setub(max(node.pd[s_o][p], 0.00))
                        model.qc_curt[i, s_m, s_o, p].setub(max(node.qd[s_o][p], 0.00))

    # - Transformers
    model.r = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=1.0)
    for i in model.branches:
        branch = network.branches[i]
        if branch.is_transformer:
            # - Transformer
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        if params.transf_reg and branch.vmag_reg:
                            model.r[i, s_m, s_o, p].setub(TRANSFORMER_MAXIMUM_RATIO)
                            model.r[i, s_m, s_o, p].setlb(TRANSFORMER_MINIMUM_RATIO)
                        else:
                            model.r[i, s_m, s_o, p].fix(branch.ratio)
        else:
            # - Line, or FACTS
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        if branch.ratio != 0.0:
                            model.r[i, s_m, s_o, p].fix(branch.ratio)            # Voltage regulation device, use given ratio
                        else:
                            model.r[i, s_m, s_o, p].fix(1.00)

    # - Energy Storage devices
    if params.es_reg:
        model.es_soc = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
        model.es_sch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_pch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
        model.es_qch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
        model.es_sdch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_pdch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
        model.es_qdch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
        if params.ess_relax:
            model.penalty_es_ch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
            model.penalty_es_dch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
            model.penalty_es_soc = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
            model.penalty_es_comp = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
            model.penalty_es_soc_day_balance = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        for e in model.energy_storages:
            energy_storage = network.energy_storages[e]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.es_soc[e, s_m, s_o, p] = energy_storage.e_init
                        model.es_soc[e, s_m, s_o, p].setlb(energy_storage.e_min)
                        model.es_soc[e, s_m, s_o, p].setub(energy_storage.e_max)
                        model.es_sch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_pch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_pch[e, s_m, s_o, p].setlb(-energy_storage.s)
                        model.es_qch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_qch[e, s_m, s_o, p].setlb(-energy_storage.s)
                        model.es_sdch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_pdch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_pdch[e, s_m, s_o, p].setlb(-energy_storage.s)
                        model.es_qdch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_qdch[e, s_m, s_o, p].setlb(-energy_storage.s)

    # - Shared Energy Storage devices
    model.shared_es_s_rated = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals)
    model.shared_es_e_rated = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals)
    model.shared_es_s_rated_fixed = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals)          # Benders' -- used to get the dual variables (sensitivities)
    model.shared_es_e_rated_fixed = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals)          # (...)
    model.shared_es_s_slack_up = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals)             # Benders' -- ensures feasibility of the subproblem
    model.shared_es_s_slack_down = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals)           # (...)
    model.shared_es_e_slack_up = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals)
    model.shared_es_e_slack_down = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals)
    model.shared_es_soc = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals)
    model.shared_es_sch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_sdch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_pch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_pdch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_qch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.shared_es_qdch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    if params.ess_relax:
        model.penalty_shared_es_ch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.penalty_shared_es_dch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.penalty_shared_es_soc = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.penalty_shared_es_comp = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.penalty_shared_es_soc_day_balance = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
    for e in model.shared_energy_storages:
        shared_energy_storage = network.shared_energy_storages[e]
        model.shared_es_s_rated_fixed[e].fix(shared_energy_storage.s)
        model.shared_es_e_rated_fixed[e].fix(shared_energy_storage.e)

    # - Expected interface power flow
    if network.is_transmission:
        model.active_distribution_networks = range(len(network.active_distribution_network_nodes))
        model.expected_interface_vmag_sqr = pe.Var(model.active_distribution_networks, model.periods, domain=pe.NonNegativeReals, initialize=1.0)
        model.expected_interface_pf_p = pe.Var(model.active_distribution_networks, model.periods, domain=pe.Reals, initialize=0.0)
        model.expected_interface_pf_q = pe.Var(model.active_distribution_networks, model.periods, domain=pe.Reals, initialize=0.0)
    else:
        model.expected_interface_vmag_sqr = pe.Var(model.periods, domain=pe.NonNegativeReals, initialize=1.0)
        model.expected_interface_pf_p = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)
        model.expected_interface_pf_q = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)

    # - Expected Shared ESS power variables
    if network.is_transmission:
        model.expected_shared_ess_p = pe.Var(model.shared_energy_storages, model.periods, domain=pe.Reals, initialize=0.0)
        model.expected_shared_ess_q = pe.Var(model.shared_energy_storages, model.periods, domain=pe.Reals, initialize=0.0)
    else:
        model.expected_shared_ess_p = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)
        model.expected_shared_ess_q = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)

    # ------------------------------------------------------------------------------------------------------------------
    # Constraints
    # - Voltage
    model.voltage_definitions = pe.ConstraintList()
    model.voltage_cons = pe.ConstraintList()
    for i in model.nodes:
        node = network.nodes[i]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:

                    # Define e_actual and f actual
                    e_actual = model.e[i, s_m, s_o, p]
                    f_actual = model.f[i, s_m, s_o, p]
                    if params.slack_voltage_limits:
                        e_actual += model.slack_e_up[i, s_m, s_o, p] - model.slack_e_down[i, s_m, s_o, p]
                        f_actual += model.slack_f_up[i, s_m, s_o, p] - model.slack_f_down[i, s_m, s_o, p]
                    model.voltage_definitions.add(model.e_actual[i, s_m, s_o, p] - e_actual <= SMALL_TOLERANCE)
                    model.voltage_definitions.add(model.e_actual[i, s_m, s_o, p] - e_actual >= -SMALL_TOLERANCE)
                    model.voltage_definitions.add(model.f_actual[i, s_m, s_o, p] - f_actual <= SMALL_TOLERANCE)
                    model.voltage_definitions.add(model.f_actual[i, s_m, s_o, p] - f_actual >= -SMALL_TOLERANCE)

                    '''
                    # Define vmag_sqr
                    vmag_sqr = model.e_actual[i, s_m, s_o, p] ** 2 + model.f_actual[i, s_m, s_o, p] ** 2
                    model.voltage_definitions.add(model.vmag_sqr[i, s_m, s_o, p] - vmag_sqr <= SMALL_TOLERANCE)
                    model.voltage_definitions.add(model.vmag_sqr[i, s_m, s_o, p] - vmag_sqr >= -SMALL_TOLERANCE)
                    '''

                    # Voltage magnitude restrictions
                    if node.type == BUS_PV:
                        if params.enforce_vg:
                            # - Enforce voltage controlled bus
                            gen_idx = network.get_gen_idx(node.bus_i)
                            vg = network.generators[gen_idx].vg
                            if not params.relaxed_model:
                                model.voltage_cons.add(model.e[i, s_m, s_o, p] ** 2 + model.f[i, s_m, s_o, p] ** 2 - vg[p] ** 2 <= SMALL_TOLERANCE)
                                model.voltage_cons.add(model.e[i, s_m, s_o, p] ** 2 + model.f[i, s_m, s_o, p] ** 2 - vg[p] ** 2 >= -SMALL_TOLERANCE)
                            else:
                                model.voltage_cons.add(model.e[i, s_m, s_o, p] ** 2 + model.f[i, s_m, s_o, p] ** 2 - vg[p] ** 2 <= model.penalty_gen_vg[i, s_m, s_o, p])
                        else:
                            ei = model.e[i, s_m, s_o, p]
                            fi = model.f[i, s_m, s_o, p]
                            model.voltage_cons.add(ei ** 2 + fi ** 2 <= node.v_max ** 2)
                            model.voltage_cons.add(ei ** 2 + fi ** 2 >= node.v_min ** 2)
                            if params.relaxed_model:
                                model.penalty_gen_vg[i, s_m, s_o, p].fix(0.00)
                    elif node.type == BUS_PQ:
                        ei = model.e[i, s_m, s_o, p]
                        fi = model.f[i, s_m, s_o, p]
                        model.voltage_cons.add(ei ** 2 + fi ** 2 <= node.v_max ** 2)
                        model.voltage_cons.add(ei ** 2 + fi ** 2 >= node.v_min ** 2)
                        if params.relaxed_model:
                            model.penalty_gen_vg[i, s_m, s_o, p].fix(0.00)

    # - Flexible Loads -- Daily energy balance
    if params.fl_reg:
        model.fl_p_balance = pe.ConstraintList()
        for i in model.nodes:
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    p_up, p_down = 0.0, 0.0
                    for p in model.periods:
                        p_up += model.flex_p_up[i, s_m, s_o, p]
                        p_down += model.flex_p_down[i, s_m, s_o, p]
                    if not params.fl_relax:
                        model.fl_p_balance.add(p_up - p_down >= -SMALL_TOLERANCE)   # FL energy balance added as a strict constraint
                        model.fl_p_balance.add(p_up - p_down <= SMALL_TOLERANCE)    # - Note: helps with convergence (numerical issues)
                    else:
                        model.fl_p_balance.add(p_up - p_down <= model.penalty_flex_day_balance[i, s_m, s_o])    # - Note: helps with convergence (numerical issues)

    # - Energy Storage constraints
    if params.es_reg:

        model.energy_storage_balance = pe.ConstraintList()
        model.energy_storage_operation = pe.ConstraintList()
        model.energy_storage_day_balance = pe.ConstraintList()
        model.energy_storage_ch_dch_exclusion = pe.ConstraintList()

        for e in model.energy_storages:

            energy_storage = network.energy_storages[e]
            soc_init = energy_storage.e_init
            soc_final = energy_storage.e_init
            eff_charge = energy_storage.eff_ch
            eff_discharge = energy_storage.eff_dch
            max_phi = acos(energy_storage.max_pf)
            min_phi = acos(energy_storage.min_pf)

            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:

                        sch = model.es_sch[e, s_m, s_o, p]
                        pch = model.es_pch[e, s_m, s_o, p]
                        qch = model.es_qch[e, s_m, s_o, p]
                        sdch = model.es_sdch[e, s_m, s_o, p]
                        pdch = model.es_pdch[e, s_m, s_o, p]
                        qdch = model.es_qdch[e, s_m, s_o, p]

                        # ESS operation
                        if not params.ess_relax:
                            model.energy_storage_operation.add(sch ** 2 - (pch ** 2 + qch ** 2) >= -(1/s_base) * SMALL_TOLERANCE)
                            model.energy_storage_operation.add(sch ** 2 - (pch ** 2 + qch ** 2) <= (1/s_base) * SMALL_TOLERANCE)
                            model.energy_storage_operation.add(sdch ** 2 - (pdch ** 2 + qdch ** 2) >= -(1/s_base) * SMALL_TOLERANCE)
                            model.energy_storage_operation.add(sdch ** 2 - (pdch ** 2 + qdch ** 2) <= (1/s_base) * SMALL_TOLERANCE)
                        else:
                            model.energy_storage_operation.add(sch ** 2 - (pch ** 2 + qch ** 2) <= model.penalty_es_ch[e, s_m, s_o, p])
                            model.energy_storage_operation.add(sdch ** 2 - (pdch ** 2 + qdch ** 2) <= model.penalty_es_dch[e, s_m, s_o, p])

                        model.energy_storage_operation.add(qch <= tan(max_phi) * pch)
                        model.energy_storage_operation.add(qch >= tan(min_phi) * pch)
                        model.energy_storage_operation.add(qdch <= tan(max_phi) * pdch)
                        model.energy_storage_operation.add(qdch >= tan(min_phi) * pdch)

                        # State-of-Charge
                        if p > 0:
                            if not params.ess_relax:
                                model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] - model.es_soc[e, s_m, s_o, p - 1] - (sch * eff_charge - sdch / eff_discharge) >= -SMALL_TOLERANCE)
                                model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] - model.es_soc[e, s_m, s_o, p - 1] - (sch * eff_charge - sdch / eff_discharge) <= SMALL_TOLERANCE)
                            else:
                                model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] - model.es_soc[e, s_m, s_o, p - 1] - (sch * eff_charge - sdch / eff_discharge) <= model.penalty_es_soc[e, s_m, s_o, p])
                        else:
                            if not params.ess_relax:
                                model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] - soc_init - (sch * eff_charge - sdch / eff_discharge) >= -(1/s_base) * SMALL_TOLERANCE)
                                model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] - soc_init - (sch * eff_charge - sdch / eff_discharge) <= (1/s_base) * SMALL_TOLERANCE)
                            else:
                                model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] - soc_init - (sch * eff_charge - sdch / eff_discharge) <= model.penalty_es_soc[e, s_m, s_o, p])

                        # Charging/discharging complementarity constraints
                        if not params.ess_relax:
                            model.energy_storage_ch_dch_exclusion.add(sch * sdch >= -(1 / s_base ** 2) * SMALL_TOLERANCE)  # Note: helps with convergence
                            model.energy_storage_ch_dch_exclusion.add(sch * sdch <= (1 / s_base ** 2) * SMALL_TOLERANCE)
                        else:
                            model.energy_storage_ch_dch_exclusion.add(sch * sdch <= model.penalty_es_comp[e, s_m, s_o, p])

                    if not params.ess_relax:
                        model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] - soc_final >= -SMALL_TOLERANCE)
                        model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] - soc_final <= SMALL_TOLERANCE)
                    else:
                        model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] - soc_final <= model.penalty_es_soc_day_balance[e, s_m, s_o])

    # - Shared Energy Storage constraints
    model.shared_energy_storage_balance = pe.ConstraintList()
    model.shared_energy_storage_operation = pe.ConstraintList()
    model.shared_energy_storage_day_balance = pe.ConstraintList()
    model.shared_energy_storage_ch_dch_exclusion = pe.ConstraintList()
    model.shared_energy_storage_s_sensitivities = pe.ConstraintList()
    model.shared_energy_storage_e_sensitivities = pe.ConstraintList()
    for e in model.shared_energy_storages:

        shared_energy_storage = network.shared_energy_storages[e]
        eff_charge = shared_energy_storage.eff_ch
        eff_discharge = shared_energy_storage.eff_dch
        max_phi = acos(shared_energy_storage.max_pf)
        min_phi = acos(shared_energy_storage.min_pf)

        s_max = model.shared_es_s_rated[e]
        soc_max = model.shared_es_e_rated[e] * ENERGY_STORAGE_MAX_ENERGY_STORED
        soc_min = model.shared_es_e_rated[e] * ENERGY_STORAGE_MIN_ENERGY_STORED
        soc_init = model.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
        soc_final = model.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:

                    sch = model.shared_es_sch[e, s_m, s_o, p]
                    sdch = model.shared_es_sdch[e, s_m, s_o, p]
                    pch = model.shared_es_pch[e, s_m, s_o, p]
                    pdch = model.shared_es_pdch[e, s_m, s_o, p]
                    qch = model.shared_es_qch[e, s_m, s_o, p]
                    qdch = model.shared_es_qdch[e, s_m, s_o, p]

                    # ESS operation
                    model.shared_energy_storage_operation.add(sch <= s_max)
                    model.shared_energy_storage_operation.add(sdch <= s_max)
                    model.shared_energy_storage_operation.add(pch <= s_max)
                    model.shared_energy_storage_operation.add(pdch <= s_max)
                    model.shared_energy_storage_operation.add(qch <= s_max)
                    model.shared_energy_storage_operation.add(qch >= -s_max)
                    model.shared_energy_storage_operation.add(qdch <= s_max)
                    model.shared_energy_storage_operation.add(qdch >= -s_max)

                    if not params.ess_relax:
                        model.shared_energy_storage_operation.add(sch ** 2 - (pch ** 2 + qch ** 2) >= -(1/s_base) * SMALL_TOLERANCE)
                        model.shared_energy_storage_operation.add(sch ** 2 - (pch ** 2 + qch ** 2) <= (1/s_base) * SMALL_TOLERANCE)
                        model.shared_energy_storage_operation.add(sdch ** 2 - (pdch ** 2 + qdch ** 2) >= -(1/s_base) * SMALL_TOLERANCE)
                        model.shared_energy_storage_operation.add(sdch ** 2 - (pdch ** 2 + qdch ** 2) <= (1/s_base) * SMALL_TOLERANCE)
                    else:
                        model.shared_energy_storage_operation.add(sch ** 2 - (pch ** 2 + qch ** 2) <= model.penalty_shared_es_ch[e, s_m, s_o, p])
                        model.shared_energy_storage_operation.add(sdch ** 2 - (pdch ** 2 + qdch ** 2) <= model.penalty_shared_es_dch[e, s_m, s_o, p])

                    model.shared_energy_storage_operation.add(qch <= tan(max_phi) * pch)
                    model.shared_energy_storage_operation.add(qch >= tan(min_phi) * pch)
                    model.shared_energy_storage_operation.add(qdch <= tan(max_phi) * pdch)
                    model.shared_energy_storage_operation.add(qdch >= tan(min_phi) * pdch)

                    model.shared_energy_storage_operation.add(model.shared_es_soc[e, s_m, s_o, p] <= soc_max)
                    model.shared_energy_storage_operation.add(model.shared_es_soc[e, s_m, s_o, p] >= soc_min)

                    # State-of-Charge
                    if p > 0:
                        if not params.ess_relax:
                            model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] - model.shared_es_soc[e, s_m, s_o, p - 1] - (sch * eff_charge - sdch / eff_discharge) >= -SMALL_TOLERANCE)
                            model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] - model.shared_es_soc[e, s_m, s_o, p - 1] - (sch * eff_charge - sdch / eff_discharge) <= SMALL_TOLERANCE)
                        else:
                            model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] - model.shared_es_soc[e, s_m, s_o, p - 1] - (sch * eff_charge - sdch / eff_discharge) <= model.penalty_shared_es_soc[e, s_m, s_o, p])
                    else:
                        if not params.ess_relax:
                            model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] - soc_init - (sch * eff_charge - sdch / eff_discharge) >= -SMALL_TOLERANCE)
                            model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] - soc_init - (sch * eff_charge - sdch / eff_discharge) <= SMALL_TOLERANCE)
                        else:
                            model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] - soc_init - (sch * eff_charge - sdch / eff_discharge) <= model.penalty_shared_es_soc[e, s_m, s_o, p])

                    # Charging/discharging complementarity constraints
                    if not params.ess_relax:
                        model.shared_energy_storage_ch_dch_exclusion.add(sch * sdch >= -(1 / s_base) * SMALL_TOLERANCE)
                        model.shared_energy_storage_ch_dch_exclusion.add(sch * sdch <= (1 / s_base) * SMALL_TOLERANCE)
                    else:
                        model.shared_energy_storage_ch_dch_exclusion.add(sch * sdch <= model.penalty_shared_es_comp[e, s_m, s_o, p])

                if not params.ess_relax:
                    model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] - soc_final >= -SMALL_TOLERANCE)
                    model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] - soc_final <= SMALL_TOLERANCE)
                else:
                    model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] - soc_final <= model.penalty_shared_es_soc_day_balance[e, s_m, s_o])

        model.shared_energy_storage_s_sensitivities.add(model.shared_es_s_rated[e] - (model.shared_es_s_slack_up[e] - model.shared_es_s_slack_down[e]) == model.shared_es_s_rated_fixed[e])
        model.shared_energy_storage_e_sensitivities.add(model.shared_es_e_rated[e] - (model.shared_es_e_slack_up[e] - model.shared_es_e_slack_down[e]) == model.shared_es_e_rated_fixed[e])

    # - Node Balance constraints
    model.node_balance_cons_p = pe.ConstraintList()
    model.node_balance_cons_q = pe.ConstraintList()
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for p in model.periods:
                for i in range(len(network.nodes)):

                    node = network.nodes[i]

                    Pd = model.pc[i, s_m, s_o, p]
                    Qd = model.qc[i, s_m, s_o, p]
                    if params.fl_reg:
                        Pd += (model.flex_p_up[i, s_m, s_o, p] - model.flex_p_down[i, s_m, s_o, p])
                    if params.l_curt:
                        Pd -= model.pc_curt[i, s_m, s_o, p]
                        Qd -= model.qc_curt[i, s_m, s_o, p]
                    if params.es_reg:
                        for e in model.energy_storages:
                            if network.energy_storages[e].bus == node.bus_i:
                                Pd += (model.es_pch[e, s_m, s_o, p] - model.es_pdch[e, s_m, s_o, p])
                                Qd += (model.es_qch[e, s_m, s_o, p] - model.es_qdch[e, s_m, s_o, p])
                    for e in model.shared_energy_storages:
                        if network.shared_energy_storages[e].bus == node.bus_i:
                            Pd += (model.shared_es_pch[e, s_m, s_o, p] - model.shared_es_pdch[e, s_m, s_o, p])
                            Qd += (model.shared_es_qch[e, s_m, s_o, p] - model.shared_es_qdch[e, s_m, s_o, p])

                    Pg = 0.0
                    Qg = 0.0
                    for g in model.generators:
                        generator = network.generators[g]
                        if generator.bus == node.bus_i:
                            Pg += model.pg[g, s_m, s_o, p]
                            if params.rg_curt:
                                Pg -= model.pg_curt[g, s_m, s_o, p]
                            Qg += model.qg[g, s_m, s_o, p]

                    ei = model.e_actual[i, s_m, s_o, p]
                    fi = model.f_actual[i, s_m, s_o, p]

                    Pi = node.gs * (ei ** 2 + fi ** 2)
                    Qi = -node.bs * (ei ** 2 + fi ** 2)
                    for b in range(len(network.branches)):
                        branch = network.branches[b]
                        if branch.fbus == node.bus_i or branch.tbus == node.bus_i:

                            rij = 1 / model.r[b, s_m, s_o, p]

                            if branch.fbus == node.bus_i:

                                fnode_idx = network.get_node_idx(branch.fbus)
                                tnode_idx = network.get_node_idx(branch.tbus)

                                ei = model.e_actual[fnode_idx, s_m, s_o, p]
                                fi = model.f_actual[fnode_idx, s_m, s_o, p]
                                ej = model.e_actual[tnode_idx, s_m, s_o, p]
                                fj = model.f_actual[tnode_idx, s_m, s_o, p]

                                Pi += branch.g * (ei ** 2 + fi ** 2) * rij**2
                                Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
                                Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2) * rij**2
                                Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))
                            else:

                                fnode_idx = network.get_node_idx(branch.tbus)
                                tnode_idx = network.get_node_idx(branch.fbus)

                                ei = model.e_actual[fnode_idx, s_m, s_o, p]
                                fi = model.f_actual[fnode_idx, s_m, s_o, p]
                                ej = model.e_actual[tnode_idx, s_m, s_o, p]
                                fj = model.f_actual[tnode_idx, s_m, s_o, p]

                                Pi += branch.g * (ei ** 2 + fi ** 2)
                                Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
                                Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2)
                                Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))

                    model.node_balance_cons_p.add(Pg - Pd - Pi >= -SMALL_TOLERANCE)
                    model.node_balance_cons_p.add(Pg - Pd - Pi <= SMALL_TOLERANCE)
                    model.node_balance_cons_q.add(Qg - Qd - Qi >= -SMALL_TOLERANCE)
                    model.node_balance_cons_q.add(Qg - Qd - Qi <= SMALL_TOLERANCE)

    # - Branch Power Flow constraints (current)
    model.branch_power_flow_cons = pe.ConstraintList()
    model.branch_power_flow_lims = pe.ConstraintList()
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for p in model.periods:
                for b in model.branches:

                    branch = network.branches[b]
                    rating = branch.rate / network.baseMVA
                    if rating == 0.0:
                        rating = BRANCH_UNKNOWN_RATING
                    fnode_idx = network.get_node_idx(branch.fbus)
                    tnode_idx = network.get_node_idx(branch.tbus)

                    ei = model.e_actual[fnode_idx, s_m, s_o, p]
                    fi = model.f_actual[fnode_idx, s_m, s_o, p]
                    ej = model.e_actual[tnode_idx, s_m, s_o, p]
                    fj = model.f_actual[tnode_idx, s_m, s_o, p]

                    iij_sqr = (branch.g**2 + branch.b**2) * ((ei - ej)**2 + (fi - fj)**2)

                    model.branch_power_flow_cons.add(model.iij_sqr[b, s_m, s_o, p] - iij_sqr >= -SMALL_TOLERANCE)
                    model.branch_power_flow_cons.add(model.iij_sqr[b, s_m, s_o, p] - iij_sqr <= SMALL_TOLERANCE)
                    model.branch_power_flow_lims.add(model.iij_sqr[b, s_m, s_o, p] <= rating**2)

                    '''
                    if params.slack_line_limits:
                        model.branch_power_flow_lims.add(model.iij_sqr[b, s_m, s_o, p] - model.slack_iij_sqr[b, s_m, s_o, p] <= rating**2)
                    else:
                        model.branch_power_flow_lims.add(model.iij_sqr[b, s_m, s_o, p] <= rating**2)
                    '''

    # - Expected Interface Power Flow (explicit definition)
    model.expected_interface_pf = pe.ConstraintList()
    model.expected_interface_voltage = pe.ConstraintList()
    if network.is_transmission:
        for dn in model.active_distribution_networks:
            node_id = network.active_distribution_network_nodes[dn]
            node_idx = network.get_node_idx(node_id)
            for p in model.periods:
                expected_pf_p = 0.0
                expected_pf_q = 0.0
                expected_vmag_sqr = 0.0
                for s_m in model.scenarios_market:
                    omega_m = network.prob_market_scenarios[s_m]
                    for s_o in model.scenarios_operation:
                        omega_o = network.prob_operation_scenarios[s_o]
                        expected_pf_p += model.pc[node_idx, s_m, s_o, p] * omega_m * omega_o
                        expected_pf_q += model.qc[node_idx, s_m, s_o, p] * omega_m * omega_o
                        #expected_vmag_sqr += model.vmag_sqr[node_idx, s_m, s_o, p] * omega_m * omega_o
                        expected_vmag_sqr += (model.e_actual[node_idx, s_m, s_o, p] ** 2 + model.f_actual[node_idx, s_m, s_o, p] ** 2) * omega_m * omega_o
                model.expected_interface_pf.add(model.expected_interface_pf_p[dn, p] - expected_pf_p >= -SMALL_TOLERANCE)       # Note: helps with convergence (numerical issues)
                model.expected_interface_pf.add(model.expected_interface_pf_p[dn, p] - expected_pf_p <= SMALL_TOLERANCE)
                model.expected_interface_pf.add(model.expected_interface_pf_q[dn, p] - expected_pf_q >= -SMALL_TOLERANCE)
                model.expected_interface_pf.add(model.expected_interface_pf_q[dn, p] - expected_pf_q <= SMALL_TOLERANCE)
                model.expected_interface_voltage.add(model.expected_interface_vmag_sqr[dn, p] - expected_vmag_sqr >= -SMALL_TOLERANCE)
                model.expected_interface_voltage.add(model.expected_interface_vmag_sqr[dn, p] - expected_vmag_sqr <= SMALL_TOLERANCE)
    else:
        ref_node_idx = network.get_node_idx(ref_node_id)
        ref_gen_idx = network.get_reference_gen_idx()
        for p in model.periods:
            expected_pf_p = 0.0
            expected_pf_q = 0.0
            expected_vmag_sqr = 0.0
            for s_m in model.scenarios_market:
                omega_m = network.prob_market_scenarios[s_m]
                for s_o in model.scenarios_operation:
                    omega_s = network.prob_operation_scenarios[s_o]
                    expected_pf_p += model.pg[ref_gen_idx, s_m, s_o, p] * omega_m * omega_s
                    expected_pf_q += model.qg[ref_gen_idx, s_m, s_o, p] * omega_m * omega_s
                    #expected_vmag_sqr += model.vmag_sqr[ref_node_idx, s_m, s_o, p] * omega_m * omega_s
                    expected_vmag_sqr += (model.e_actual[ref_node_idx, s_m, s_o, p] ** 2 + model.f_actual[ref_node_idx, s_m, s_o, p] ** 2) * omega_m * omega_s
            model.expected_interface_pf.add(model.expected_interface_pf_p[p] - expected_pf_p >= -SMALL_TOLERANCE)
            model.expected_interface_pf.add(model.expected_interface_pf_p[p] - expected_pf_p <= SMALL_TOLERANCE)
            model.expected_interface_pf.add(model.expected_interface_pf_q[p] - expected_pf_q >= -SMALL_TOLERANCE)
            model.expected_interface_pf.add(model.expected_interface_pf_q[p] - expected_pf_q <= SMALL_TOLERANCE)
            model.expected_interface_voltage.add(model.expected_interface_vmag_sqr[p] - expected_vmag_sqr >= -SMALL_TOLERANCE)
            model.expected_interface_voltage.add(model.expected_interface_vmag_sqr[p] - expected_vmag_sqr <= SMALL_TOLERANCE)

    # - Expected Shared ESS Power (explicit definition)
    if len(network.shared_energy_storages) > 0:
        model.expected_shared_ess_power = pe.ConstraintList()
        if network.is_transmission:
            for e in model.shared_energy_storages:
                for p in model.periods:
                    expected_sess_p = 0.0
                    expected_sess_q = 0.0
                    for s_m in model.scenarios_market:
                        omega_m = network.prob_market_scenarios[s_m]
                        for s_o in model.scenarios_operation:
                            omega_o = network.prob_operation_scenarios[s_o]
                            pch = model.shared_es_pch[e, s_m, s_o, p]
                            pdch = model.shared_es_pdch[e, s_m, s_o, p]
                            qch = model.shared_es_qch[e, s_m, s_o, p]
                            qdch = model.shared_es_qdch[e, s_m, s_o, p]
                            expected_sess_p += (pch - pdch) * omega_m * omega_o
                            expected_sess_q += (qch - qdch) * omega_m * omega_o
                    model.expected_shared_ess_power.add(model.expected_shared_ess_p[e, p] - expected_sess_p >= -SMALL_TOLERANCE)
                    model.expected_shared_ess_power.add(model.expected_shared_ess_p[e, p] - expected_sess_p <= SMALL_TOLERANCE)
                    model.expected_shared_ess_power.add(model.expected_shared_ess_q[e, p] - expected_sess_q >= -SMALL_TOLERANCE)
                    model.expected_shared_ess_power.add(model.expected_shared_ess_q[e, p] - expected_sess_q <= SMALL_TOLERANCE)
        else:
            shared_ess_idx = network.get_shared_energy_storage_idx(ref_node_id)
            for p in model.periods:
                expected_sess_p = 0.0
                expected_sess_q = 0.0
                for s_m in model.scenarios_market:
                    omega_m = network.prob_market_scenarios[s_m]
                    for s_o in model.scenarios_operation:
                        omega_s = network.prob_operation_scenarios[s_o]
                        pch = model.shared_es_pch[shared_ess_idx, s_m, s_o, p]
                        pdch = model.shared_es_pdch[shared_ess_idx, s_m, s_o, p]
                        qch = model.shared_es_qch[shared_ess_idx, s_m, s_o, p]
                        qdch = model.shared_es_qdch[shared_ess_idx, s_m, s_o, p]
                        expected_sess_p += (pch - pdch) * omega_m * omega_s
                        expected_sess_q += (qch - qdch) * omega_m * omega_s
                model.expected_shared_ess_power.add(model.expected_shared_ess_p[p] - expected_sess_p >= -SMALL_TOLERANCE)
                model.expected_shared_ess_power.add(model.expected_shared_ess_p[p] - expected_sess_p <= SMALL_TOLERANCE)
                model.expected_shared_ess_power.add(model.expected_shared_ess_q[p] - expected_sess_q >= -SMALL_TOLERANCE)
                model.expected_shared_ess_power.add(model.expected_shared_ess_q[p] - expected_sess_q <= SMALL_TOLERANCE)

    # ------------------------------------------------------------------------------------------------------------------
    # Objective Function
    obj = 0.0
    if params.obj_type == OBJ_MIN_COST:

        # Cost minimization
        c_p = network.cost_energy_p
        for s_m in model.scenarios_market:
            omega_market = network.prob_market_scenarios[s_m]
            for s_o in model.scenarios_operation:

                obj_scenario = 0.0
                omega_oper = network.prob_operation_scenarios[s_o]

                # Generation -- paid at market price (energy)
                for g in model.generators:
                    if network.generators[g].is_controllable():
                        for p in model.periods:
                            pg = model.pg[g, s_m, s_o, p]
                            obj_scenario += c_p[s_m][p] * network.baseMVA * pg

                # Demand side flexibility
                if params.fl_reg:
                    for i in model.nodes:
                        node = network.nodes[i]
                        for p in model.periods:
                            cost_flex = node.flexibility.cost[p]
                            flex_p_down = model.flex_p_up[i, s_m, s_o, p]
                            obj_scenario += cost_flex * network.baseMVA * (flex_p_down)

                # Load curtailment
                if params.l_curt:
                    for i in model.nodes:
                        for p in model.periods:
                            pc_curt = model.pc_curt[i, s_m, s_o, p]
                            qc_curt = model.qc_curt[i, s_m, s_o, p]
                            obj_scenario += COST_CONSUMPTION_CURTAILMENT * network.baseMVA * (pc_curt)
                            obj_scenario += COST_CONSUMPTION_CURTAILMENT * network.baseMVA * (qc_curt)

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = model.pg_curt[g, s_m, s_o, p]
                            obj_scenario += COST_GENERATION_CURTAILMENT * network.baseMVA * pg_curt

                # Voltage slacks
                if params.slack_voltage_limits:
                    for i in model.nodes:
                        for p in model.periods:
                            slack_e = model.slack_e_up[i, s_m, s_o, p] + model.slack_e_down[i, s_m, s_o, p]
                            slack_f = model.slack_f_up[i, s_m, s_o, p] + model.slack_f_down[i, s_m, s_o, p]
                            obj_scenario += COST_SLACK_VOLTAGE * network.baseMVA * (slack_e + slack_f)

                # Branch power flow slacks
                if params.slack_line_limits:
                    for b in model.branches:
                        for p in model.periods:
                            slack_iij_sqr = model.slack_iij_sqr[b, s_m, s_o, p]
                            obj_scenario += COST_SLACK_BRANCH_FLOW * network.baseMVA * slack_iij_sqr

                # Relaxation penalties
                if params.ess_relax:

                    # - Shared ESS
                    for e in model.shared_energy_storages:
                        for p in model.periods:
                            penalty_ch = model.penalty_shared_es_ch[e, s_m, s_o, p]
                            penalty_dch = model.penalty_shared_es_ch[e, s_m, s_o, p]
                            penalty_soc = model.penalty_shared_es_soc[e, s_m, s_o, p]
                            penalty_comp = model.penalty_shared_es_comp[e, s_m, s_o, p]
                            obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * (penalty_ch + penalty_dch + penalty_soc + penalty_comp)
                        penalty_day_balance = model.penalty_shared_es_soc_day_balance[e, s_m, s_o]
                        obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * penalty_day_balance

                    # - ESS
                    if params.es_reg:
                        for e in model.energy_storages:
                            for p in model.periods:
                                penalty_ch = model.penalty_es_ch[e, s_m, s_o, p]
                                penalty_dch = model.penalty_es_ch[e, s_m, s_o, p]
                                penalty_soc = model.penalty_es_soc[e, s_m, s_o, p]
                                penalty_comp = model.penalty_es_comp[e, s_m, s_o, p]
                                obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * (penalty_ch + penalty_dch + penalty_soc + penalty_comp)
                            penalty_day_balance = model.penalty_shared_es_soc_day_balance[e, s_m, s_o]
                            obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * penalty_day_balance

                # - Flexibility day balance
                if params.fl_reg:
                    if params.fl_relax:
                        for i in model.nodes:
                            penalty_day_balance = model.penalty_flex_day_balance[i, s_m, s_o]
                            obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * penalty_day_balance

                # - PV bus voltage set-point
                if params.relaxed_model:
                    for i in model.nodes:
                        for p in model.periods:
                            penalty_gen_vg = model.penalty_gen_vg[i, s_m, s_o, p]
                            obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * (penalty_gen_vg)

                obj += obj_scenario * omega_market * omega_oper

        # ESS Planning -- Slack Penalty
        for e in model.shared_energy_storages:
            obj += PENALTY_ESS_SLACK * (model.shared_es_s_slack_up[e] + model.shared_es_s_slack_down[e])
            obj += PENALTY_ESS_SLACK * (model.shared_es_e_slack_up[e] + model.shared_es_e_slack_down[e])

        model.objective = pe.Objective(sense=pe.minimize, expr=obj)
    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:

        # Congestion Management
        for s_m in model.scenarios_market:

            omega_market = network.prob_market_scenarios[s_m]

            for s_o in model.scenarios_operation:

                obj_scenario = 0.0
                omega_oper = network.prob_operation_scenarios[s_o]

                # Branch power flow slacks
                if params.slack_line_limits:
                    for k in model.branches:
                        for p in model.periods:
                            slack_i_sqr = model.slack_iij_sqr[k, s_m, s_o, p]
                            obj_scenario += PENALTY_SLACK_BRANCH_FLOW * slack_i_sqr

                # Voltage slacks
                if params.slack_voltage_limits:
                    for i in model.nodes:
                        for p in model.periods:
                            slack_e = model.slack_e_up[i, s_m, s_o, p] + model.slack_e_down[i, s_m, s_o, p]
                            slack_f = model.slack_f_up[i, s_m, s_o, p] + model.slack_f_down[i, s_m, s_o, p]
                            obj_scenario += PENALTY_SLACK_VOLTAGE * (slack_e + slack_f)

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = model.pg_curt[g, s_m, s_o, p]
                            obj_scenario += PENALTY_GENERATION_CURTAILMENT * pg_curt

                # Consumption curtailment
                if params.l_curt:
                    for i in model.nodes:
                        for p in model.periods:
                            pc_curt = model.pc_curt[i, s_m, s_o, p]
                            qc_curt = model.qc_curt[i, s_m, s_o, p]
                            obj_scenario += PENALTY_LOAD_CURTAILMENT * pc_curt
                            obj_scenario += PENALTY_LOAD_CURTAILMENT * qc_curt

                # Relaxation penalties
                if params.ess_relax:

                    # - Shared ESS
                    for e in model.shared_energy_storages:
                        for p in model.periods:
                            penalty_ch = model.penalty_shared_es_ch[e, s_m, s_o, p]
                            penalty_dch = model.penalty_shared_es_ch[e, s_m, s_o, p]
                            penalty_soc = model.penalty_shared_es_soc[e, s_m, s_o, p]
                            penalty_comp = model.penalty_shared_es_comp[e, s_m, s_o, p]
                            obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * (penalty_ch + penalty_dch + penalty_soc + penalty_comp)
                        penalty_day_balance = model.penalty_shared_es_soc_day_balance[e, s_m, s_o]
                        obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * penalty_day_balance

                    # - ESS
                    if params.es_reg:
                        for e in model.energy_storages:
                            for p in model.periods:
                                penalty_ch = model.penalty_es_ch[e, s_m, s_o, p]
                                penalty_dch = model.penalty_es_ch[e, s_m, s_o, p]
                                penalty_soc = model.penalty_es_soc[e, s_m, s_o, p]
                                penalty_comp = model.penalty_es_comp[e, s_m, s_o, p]
                                obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * (penalty_ch + penalty_dch + penalty_soc + penalty_comp)
                            penalty_day_balance = model.penalty_shared_es_soc_day_balance[e, s_m, s_o]
                            obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * penalty_day_balance

                # Flexibility day balance
                if params.fl_reg:
                    if params.fl_relax:
                        for i in model.nodes:
                            penalty_day_balance = model.penalty_flex_day_balance[i, s_m, s_o]
                            obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * penalty_day_balance

                # PV bus voltage set-point
                if params.relaxed_model:
                    for i in model.nodes:
                        for p in model.periods:
                            penalty_gen_vg = model.penalty_gen_vg[i, s_m, s_o, p]
                            obj_scenario += PENALTY_RELAXED_MODEL * network.baseMVA * (penalty_gen_vg)

                obj += obj_scenario * omega_market * omega_oper

        # ESS Planning -- Slack Penalty
        for e in model.shared_energy_storages:
            obj += PENALTY_ESS_SLACK * (model.shared_es_s_slack_up[e] + model.shared_es_s_slack_down[e])
            obj += PENALTY_ESS_SLACK * (model.shared_es_e_slack_up[e] + model.shared_es_e_slack_down[e])

        model.objective = pe.Objective(sense=pe.minimize, expr=obj)
    else:
        print(f'[ERROR] Unrecognized or invalid objective. Objective = {params.obj_type}. Exiting...')
        exit(ERROR_NETWORK_MODEL)

    # Model suffixes (used for warm start)
    model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)  # Ipopt bound multipliers (obtained from solution)
    model.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.ipopt_zL_in = pe.Suffix(direction=pe.Suffix.EXPORT)  # Ipopt bound multipliers (sent to solver)
    model.ipopt_zU_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)  # Obtain dual solutions from previous solve and send to warm start

    return model


def _run_smopf(network, model, params, from_warm_start=False):

    solver = po.SolverFactory(params.solver_params.solver, executable=params.solver_params.solver_path)

    if from_warm_start:
        model.ipopt_zL_in.update(model.ipopt_zL_out)
        model.ipopt_zU_in.update(model.ipopt_zU_out)
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['warm_start_bound_push'] = 1e-9
        solver.options['warm_start_mult_bound_push'] = 1e-9
        solver.options['mu_init'] = 1e-9

    if params.solver_params.verbose:
        solver.options['print_level'] = 6
        solver.options['output_file'] = 'optim_log.txt'

    if params.solver_params.solver == 'ipopt':
        solver.options['tol'] = params.solver_params.solver_tol
        solver.options['acceptable_tol'] = params.solver_params.solver_tol * 1e3
        solver.options['acceptable_iter'] = 5
        #solver.options['nlp_scaling_method'] = 'none'
        solver.options['max_iter'] = 10000
        solver.options['linear_solver'] = params.solver_params.linear_solver

    result = solver.solve(model, tee=params.solver_params.verbose)

    '''
    import logging
    from pyomo.util.infeasible import log_infeasible_constraints
    filename = os.path.join(os.getcwd(), 'example.log')
    print(log_infeasible_constraints(model, log_expression=True, log_variables=True))
    #logging.basicConfig(filename=filename, encoding='utf-8', level=logging.INFO)
    '''

    return result


# ======================================================================================================================
#   NETWORK read functions -- JSON format
# ======================================================================================================================
def _read_network_from_json_file(network, filename):

    network_data = convert_json_to_dict(read_json_file(filename))

    # Network base
    network.baseMVA = float(network_data['baseMVA'])

    # Nodes
    for node_data in network_data['bus']:
        node = Node()
        node.bus_i = int(node_data['bus_i'])
        node.type = int(node_data['type'])
        node.gs = float(node_data['Gs']) / network.baseMVA
        node.bs = float(node_data['Bs']) / network.baseMVA
        node.base_kv = float(node_data['baseKV'])
        node.v_max = float(node_data['Vmax'])
        node.v_min = float(node_data['Vmin'])
        network.nodes.append(node)

    # Generators
    for gen_data in network_data['gen']:
        generator = Generator()
        generator.gen_id = int(gen_data['gen_id'])
        generator.bus = int(gen_data['bus'])
        if not network.node_exists(generator.bus):
            print(f'[ERROR] Generator {generator.gen_id}. Node {generator.bus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        generator.pmax = float(gen_data['Pmax']) / network.baseMVA
        generator.pmin = float(gen_data['Pmin']) / network.baseMVA
        generator.qmax = float(gen_data['Qmax']) / network.baseMVA
        generator.qmin = float(gen_data['Qmin']) / network.baseMVA
        generator.vg = float(gen_data['Vg'])
        generator.status = float(gen_data['status'])
        gen_type = gen_data['type']
        if gen_type == 'REF':
            generator.gen_type = GEN_REFERENCE
        elif gen_type == 'CONV':
            generator.gen_type = GEN_CONV
        elif gen_type == 'PV':
            generator.gen_type = GEN_RES_SOLAR
        elif gen_type == 'WIND':
            generator.gen_type = GEN_RES_WIND
        elif gen_type == 'RES_OTHER':
            generator.gen_type = GEN_RES_OTHER
        elif gen_type == 'RES_CONTROLLABLE':
            generator.gen_type = GEN_RES_CONTROLLABLE
        network.generators.append(generator)

    # Lines
    for line_data in network_data['line']:
        branch = Branch()
        branch.branch_id = int(line_data['branch_id'])
        branch.fbus = int(line_data['fbus'])
        if not network.node_exists(branch.fbus):
            print(f'[ERROR] Line {branch.branch_id }. Node {branch.fbus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        branch.tbus = int(line_data['tbus'])
        if not network.node_exists(branch.tbus):
            print(f'[ERROR] Line {branch.branch_id }. Node {branch.tbus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        branch.r = float(line_data['r'])
        branch.x = float(line_data['x'])
        branch.b_sh = float(line_data['b'])
        branch.rate = float(line_data['rating'])
        branch.status = int(line_data['status'])
        network.branches.append(branch)

    # Transformers
    if 'transformer' in network_data:
        for transf_data in network_data['transformer']:
            branch = Branch()
            branch.branch_id = int(transf_data['branch_id'])
            branch.fbus = int(transf_data['fbus'])
            if not network.node_exists(branch.fbus):
                print(f'[ERROR] Transformer {branch.branch_id}. Node {branch.fbus} does not exist! Exiting...')
                exit(ERROR_NETWORK_FILE)
            branch.tbus = int(transf_data['tbus'])
            if not network.node_exists(branch.tbus):
                print(f'[ERROR] Transformer {branch.branch_id}. Node {branch.tbus} does not exist! Exiting...')
                exit(ERROR_NETWORK_FILE)
            branch.r = float(transf_data['r'])
            branch.x = float(transf_data['x'])
            branch.b_sh = float(transf_data['b'])
            branch.rate = float(transf_data['rating'])
            branch.ratio = float(transf_data['ratio'])
            branch.status = bool(transf_data['status'])
            branch.is_transformer = True
            branch.vmag_reg = bool(transf_data['vmag_reg'])
            network.branches.append(branch)

    # Energy Storages
    if 'energy_storage' in network_data:
        for energy_storage_data in network_data['energy_storage']:
            energy_storage = EnergyStorage()
            energy_storage.es_id = int(energy_storage_data['es_id'])
            energy_storage.bus = int(energy_storage_data['bus'])
            if not network.node_exists(energy_storage.bus):
                print(f'[ERROR] Energy Storage {energy_storage.es_id}. Node {energy_storage.bus} does not exist! Exiting...')
                exit(ERROR_NETWORK_FILE)
            energy_storage.s = float(energy_storage_data['s']) / network.baseMVA
            energy_storage.e = float(energy_storage_data['e']) / network.baseMVA
            energy_storage.e_init = float(energy_storage_data['e_init']) / network.baseMVA
            energy_storage.e_min = float(energy_storage_data['e_min']) / network.baseMVA
            energy_storage.e_max = float(energy_storage_data['e_max']) / network.baseMVA
            energy_storage.eff_ch = float(energy_storage_data['eff_ch'])
            energy_storage.eff_dch = float(energy_storage_data['eff_dch'])
            energy_storage.max_pf = float(energy_storage_data['max_pf'])
            energy_storage.min_pf = float(energy_storage_data['min_pf'])
            network.energy_storages.append(energy_storage)


# ======================================================================================================================
#   NETWORK OPERATIONAL DATA read functions
# ======================================================================================================================
def _read_network_operational_data_from_file(network, filename):

    data = {
        'consumption': {
            'pc': dict(), 'qc': dict()
        },
        'flexibility': {
            'upward': dict(),
            'downward': dict(),
            'cost': dict()
        },
        'generation': {
            'pg': dict(), 'qg': dict(), 'status': list()
        }
    }

    # Scenario information
    num_gen_cons_scenarios, prob_gen_cons_scenarios = _get_operational_scenario_info_from_excel_file(filename, 'Main')
    network.prob_operation_scenarios = prob_gen_cons_scenarios

    # Consumption and Generation data -- by scenario
    for i in range(len(network.prob_operation_scenarios)):

        sheet_name_pc = f'Pc, {network.day}, S{i + 1}'
        sheet_name_qc = f'Qc, {network.day}, S{i + 1}'
        sheet_name_pg = f'Pg, {network.day}, S{i + 1}'
        sheet_name_qg = f'Qg, {network.day}, S{i + 1}'

        # Consumption per scenario (active, reactive power)
        pc_scenario = _get_consumption_flexibility_data_from_excel_file(filename, sheet_name_pc)
        qc_scenario = _get_consumption_flexibility_data_from_excel_file(filename, sheet_name_qc)
        if not pc_scenario:
            print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. No active power consumption data provided for scenario {i + 1}. Exiting...')
            exit(ERROR_OPERATIONAL_DATA_FILE)
        if not qc_scenario:
            print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. No reactive power consumption data provided for scenario {i + 1}. Exiting...')
            exit(ERROR_OPERATIONAL_DATA_FILE)
        data['consumption']['pc'][i] = pc_scenario
        data['consumption']['qc'][i] = qc_scenario

        # Generation per scenario (active, reactive power)
        num_renewable_gens = network.get_num_renewable_gens()
        if num_renewable_gens > 0:
            pg_scenario = _get_generation_data_from_excel_file(filename, sheet_name_pg)
            qg_scenario = _get_generation_data_from_excel_file(filename, sheet_name_qg)
            if not pg_scenario:
                print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. No active power generation data provided for scenario {i + 1}. Exiting...')
                exit(ERROR_OPERATIONAL_DATA_FILE)
            if not qg_scenario:
                print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. No reactive power generation data provided for scenario {i + 1}. Exiting...')
                exit(ERROR_OPERATIONAL_DATA_FILE)
            data['generation']['pg'][i] = pg_scenario
            data['generation']['qg'][i] = qg_scenario

    # Generators status. Note: common to all scenarios
    gen_status = _get_generator_status_from_excel_file(filename, f'GenStatus, {network.day}')
    if not gen_status:
        for g in range(len(network.generators)):
            gen_status.append([network.generators[g].status for _ in range(network.num_instants)])
    data['generation']['status'] = gen_status

    # Flexibility data
    flex_up_p = _get_consumption_flexibility_data_from_excel_file(filename, f'UpFlex, {network.day}')
    if not flex_up_p:
        for node in network.nodes:
            flex_up_p[node.bus_i] = [0.0 for _ in range(network.num_instants)]
    data['flexibility']['upward'] = flex_up_p

    flex_down_p = _get_consumption_flexibility_data_from_excel_file(filename, f'DownFlex, {network.day}')
    if not flex_down_p:
        for node in network.nodes:
            flex_down_p[node.bus_i] = [0.0 for _ in range(network.num_instants)]
    data['flexibility']['downward'] = flex_down_p

    flex_cost = _get_consumption_flexibility_data_from_excel_file(filename, f'CostFlex, {network.day}')
    if not flex_cost:
        for node in network.nodes:
            flex_cost[node.bus_i] = [0.0 for _ in range(network.num_instants)]
    data['flexibility']['cost'] = flex_cost

    return data


def _get_operational_scenario_info_from_excel_file(filename, sheet_name):

    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        if is_int(df.iloc[0, 1]):
            num_scenarios = int(df.iloc[0, 1])
        prob_scenarios = list()
        for i in range(num_scenarios):
            if is_number(df.iloc[0, i+2]):
                prob_scenarios.append(float(df.iloc[0, i+2]))
    except:
        print('[ERROR] Workbook {}. Sheet {} does not exist.'.format(filename, sheet_name))
        exit(1)

    if num_scenarios != len(prob_scenarios):
        print('[WARNING] Workbook {}. Data file. Number of scenarios different from the probability vector!'.format(filename))

    if round(sum(prob_scenarios), 2) != 1.00:
        print('[ERROR] Workbook {}. Probability of scenarios does not add up to 100%.'.format(filename))
        exit(ERROR_OPERATIONAL_DATA_FILE)

    return num_scenarios, prob_scenarios


def _get_consumption_flexibility_data_from_excel_file(filename, sheet_name):

    try:
        data = pd.read_excel(filename, sheet_name=sheet_name)
        num_rows, num_cols = data.shape
        processed_data = dict()
        for i in range(num_rows):
            node_id = data.iloc[i, 0]
            processed_data[node_id] = [0.0 for _ in range(num_cols - 1)]
        for node_id in processed_data:
            node_values = [0.0 for _ in range(num_cols - 1)]
            for i in range(0, num_rows):
                aux_node_id = data.iloc[i, 0]
                if aux_node_id == node_id:
                    for j in range(0, num_cols - 1):
                        node_values[j] += data.iloc[i, j + 1]
            processed_data[node_id] = node_values
    except:
        print(f'[WARNING] Workbook {filename}. Sheet {sheet_name} does not exist.')
        processed_data = {}

    return processed_data


def _get_generation_data_from_excel_file(filename, sheet_name):

    try:
        data = pd.read_excel(filename, sheet_name=sheet_name)
        num_rows, num_cols = data.shape
        processed_data = dict()
        for i in range(num_rows):
            gen_id = data.iloc[i, 0]
            processed_data[gen_id] = [0.0 for _ in range(num_cols - 1)]
        for gen_id in processed_data:
            processed_data_gen = [0.0 for _ in range(num_cols - 1)]
            for i in range(0, num_rows):
                aux_node_id = data.iloc[i, 0]
                if aux_node_id == gen_id:
                    for j in range(0, num_cols - 1):
                        processed_data_gen[j] += data.iloc[i, j + 1]
            processed_data[gen_id] = processed_data_gen
    except:
        print(f'[WARNING] Workbook {filename}. Sheet {sheet_name} does not exist.')
        processed_data = {}

    return processed_data


def _get_generator_status_from_excel_file(filename, sheet_name):

    try:
        data = pd.read_excel(filename, sheet_name=sheet_name)
        num_rows, num_cols = data.shape
        status_values = dict()
        for i in range(num_rows):
            gen_id = data.iloc[i, 0]
            status_values[gen_id] = [0 for _ in range(num_cols - 1)]
        for node_id in status_values:
            status_values_gen = [0 for _ in range(num_cols - 1)]
            for i in range(0, num_rows):
                aux_node_id = data.iloc[i, 0]
                if aux_node_id == node_id:
                    for j in range(0, num_cols - 1):
                        status_values_gen[j] += data.iloc[i, j + 1]
            status_values[node_id] = status_values_gen
    except:
        print(f'[WARNING] Workbook {filename}. Sheet {sheet_name} does not exist.')
        status_values = list()

    return status_values


def _update_network_with_excel_data(network, data):

    for node in network.nodes:

        node_id = node.bus_i
        node.pd = dict()         # Note: Changes Pd and Qd fields to dicts (per scenario)
        node.qd = dict()

        for s in range(len(network.prob_operation_scenarios)):
            pc = _get_consumption_from_data(data, node_id, network.num_instants, s, DATA_ACTIVE_POWER)
            qc = _get_consumption_from_data(data, node_id, network.num_instants, s, DATA_REACTIVE_POWER)
            node.pd[s] = [instant / network.baseMVA for instant in pc]
            node.qd[s] = [instant / network.baseMVA for instant in qc]
        flex_up_p = _get_flexibility_from_data(data, node_id, network.num_instants, DATA_UPWARD_FLEXIBILITY)
        flex_down_p = _get_flexibility_from_data(data, node_id, network.num_instants, DATA_DOWNWARD_FLEXIBILITY)
        flex_cost = _get_flexibility_from_data(data, node_id, network.num_instants, DATA_COST_FLEXIBILITY)
        node.flexibility.upward = [p / network.baseMVA for p in flex_up_p]
        node.flexibility.downward = [q / network.baseMVA for q in flex_down_p]
        node.flexibility.cost = flex_cost

    for generator in network.generators:

        generator.pg = dict()  # Note: Changes Pg and Qg fields to dicts (per scenario)
        generator.qg = dict()

        # Active and Reactive power
        for s in range(len(network.prob_operation_scenarios)):
            if generator.gen_type in GEN_CURTAILLABLE_TYPES:
                pg = _get_generation_from_data(data, generator.gen_id, s, DATA_ACTIVE_POWER)
                qg = _get_generation_from_data(data, generator.gen_id, s, DATA_REACTIVE_POWER)
                generator.pg[s] = [instant / network.baseMVA for instant in pg]
                generator.qg[s] = [instant / network.baseMVA for instant in qg]
            else:
                generator.pg[s] = [0.00 for _ in range(network.num_instants)]
                generator.qg[s] = [0.00 for _ in range(network.num_instants)]

        # Status
        generator.status = data['generation']['status'][generator.gen_id]

    network.data_loaded = True


def _get_consumption_from_data(data, node_id, num_instants, idx_scenario, type):

    if type == DATA_ACTIVE_POWER:
        power_label = 'pc'
    else:
        power_label = 'qc'

    for node in data['consumption'][power_label][idx_scenario]:
        if node == node_id:
            return data['consumption'][power_label][idx_scenario][node_id]

    consumption = [0.0 for _ in range(num_instants)]

    return consumption


def _get_flexibility_from_data(data, node_id, num_instants, flex_type):

    if flex_type == DATA_UPWARD_FLEXIBILITY:
        flex_label = 'upward'
    elif flex_type == DATA_DOWNWARD_FLEXIBILITY:
        flex_label = 'downward'
    elif flex_type == DATA_COST_FLEXIBILITY:
        flex_label = 'cost'
    else:
        print('[ERROR] Unrecognized flexibility type in get_flexibility_from_data. Exiting.')
        exit(1)

    for node in data['flexibility'][flex_label]:
        if node == node_id:
            return data['flexibility'][flex_label][node_id]

    flex = [0.0 for _ in range(num_instants)]   # Returns empty flexibility vector

    return flex


def _get_generation_from_data(data, gen_id, idx_scenario, type):

    if type == DATA_ACTIVE_POWER:
        power_label = 'pg'
    else:
        power_label = 'qg'

    return data['generation'][power_label][idx_scenario][gen_id]


# ======================================================================================================================
#   NETWORK diagram functions (plot)
# ======================================================================================================================
def _plot_networkx_diagram(network, data_dir='data'):

    node_labels = {}
    node_voltage_labels = {}
    node_colors = ['lightblue' for _ in network.nodes]

    # Aux - Encapsulated Branch list
    branches = []
    edge_labels = {}
    line_list, open_line_list = [], []
    transf_list, open_transf_list = [], []
    for branch in network.branches:
        if branch.is_transformer:
            branches.append({'type': 'transformer', 'data': branch})
        else:
            branches.append({'type': 'line', 'data': branch})

    # Build graph
    graph = nx.Graph()
    for i in range(len(network.nodes)):
        node = network.nodes[i]
        graph.add_node(node.bus_i)
        node_labels[node.bus_i] = '{}'.format(node.bus_i)
        node_voltage_labels[node.bus_i] = '{} kV'.format(node.base_kv)
        if node.type == BUS_REF:
            node_colors[i] = 'red'
        elif node.type == BUS_PV:
            node_colors[i] = 'green'
        elif network.has_energy_storage_device(node.bus_i):
            node_colors[i] = 'blue'

    for i in range(len(branches)):
        branch = branches[i]
        if branch['type'] == 'line':
            graph.add_edge(branch['data'].fbus, branch['data'].tbus)
            if branch['data'].status == 1:
                line_list.append((branch['data'].fbus, branch['data'].tbus))
            else:
                open_line_list.append((branch['data'].fbus, branch['data'].tbus))
        if branch['type'] == 'transformer':
            graph.add_edge(branch['data'].fbus, branch['data'].tbus)
            if branch['data'].status == 1:
                transf_list.append((branch['data'].fbus, branch['data'].tbus))
            else:
                open_transf_list.append((branch['data'].fbus, branch['data'].tbus))
            ratio = '{:.3f}'.format(branch['data'].ratio)
            edge_labels[(branch['data'].fbus, branch['data'].tbus)] = f'1:{ratio}'

    # Plot - coordinates
    pos = nx.spring_layout(graph)
    pos_above, pos_below = {}, {}
    for k, v in pos.items():
        pos_above[k] = (v[0], v[1] + 0.050)
        pos_below[k] = (v[0], v[1] - 0.050)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(graph, ax=ax, pos=pos, node_color=node_colors, node_size=200)
    nx.draw_networkx_labels(graph, ax=ax, pos=pos, labels=node_labels, font_size=10)
    nx.draw_networkx_labels(graph, ax=ax, pos=pos_below, labels=node_voltage_labels, font_size=5)
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=line_list, width=1.00, edge_color='black')
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=transf_list, width=1.50, edge_color='blue')
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=open_line_list, style='dashed', width=1.00, edge_color='red')
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=open_transf_list, style='dashed', width=1.50, edge_color='red')
    nx.draw_networkx_edge_labels(graph, ax=ax, pos=pos, edge_labels=edge_labels, font_size=5, rotate=False)
    plt.axis('off')

    filename = os.path.join(network.diagrams_dir, f'{network.name}_{network.year}_{network.day}.pdf')
    plt.savefig(filename, bbox_inches='tight')

    filename = os.path.join(network.diagrams_dir, f'{network.name}_{network.year}_{network.day}.png')
    plt.savefig(filename, bbox_inches='tight')


# ======================================================================================================================
#   NETWORK RESULTS functions
# ======================================================================================================================
def _process_results(network, model, params, results=dict()):

    processed_results = dict()
    processed_results['obj'] = _compute_objective_function_value(network, model, params)
    processed_results['gen_cost'] = _compute_generation_cost(network, model)
    processed_results['total_load'] = _compute_total_load(network, model, params)
    processed_results['total_gen'] = _compute_total_generation(network, model, params)
    processed_results['total_conventional_gen'] = _compute_conventional_generation(network, model, params)
    processed_results['total_renewable_gen'] = _compute_renewable_generation(network, model, params)
    processed_results['losses'] = _compute_losses(network, model, params)
    processed_results['gen_curt'] = _compute_generation_curtailment(network, model, params)
    processed_results['load_curt'] = _compute_load_curtailment(network, model, params)
    processed_results['flex_used'] = _compute_flexibility_used(network, model, params)
    if results:
        processed_results['runtime'] = float(_get_info_from_results(results, 'Time:').strip()),
    processed_results['scenarios'] = dict()

    for s_m in model.scenarios_market:

        processed_results['scenarios'][s_m] = dict()

        for s_o in model.scenarios_operation:

            processed_results['scenarios'][s_m][s_o] = {
                'voltage': {'vmag': {}, 'vang': {}},
                'consumption': {'pc': {}, 'qc': {}, 'pc_net': {}, 'qc_net': {}},
                'generation': {'pg': {}, 'qg': {}, 'pg_net': {}},
                'branches': {'power_flow': {'pij': {}, 'pji': {}, 'qij': {}, 'qji': {}, 'sij': {}, 'sji': {}},
                             'current_perc': {}, 'losses': {}, 'ratio': {}},
                'energy_storages': {'p': {}, 'q': {}, 's': {}, 'soc': {}, 'soc_percent': {}},
                'shared_energy_storages': {'p': {}, 'q': {}, 's': {}, 'soc': {}, 'soc_percent': {}}
            }

            if params.transf_reg:
                processed_results['scenarios'][s_m][s_o]['branches']['ratio'] = dict()

            if params.fl_reg:
                processed_results['scenarios'][s_m][s_o]['consumption']['p_up'] = dict()
                processed_results['scenarios'][s_m][s_o]['consumption']['p_down'] = dict()

            if params.l_curt:
                processed_results['scenarios'][s_m][s_o]['consumption']['pc_curt'] = dict()
                processed_results['scenarios'][s_m][s_o]['consumption']['qc_curt'] = dict()

            if params.rg_curt:
                processed_results['scenarios'][s_m][s_o]['generation']['pg_curt'] = dict()

            if params.es_reg:
                processed_results['scenarios'][s_m][s_o]['energy_storages']['p'] = dict()
                processed_results['scenarios'][s_m][s_o]['energy_storages']['q'] = dict()
                processed_results['scenarios'][s_m][s_o]['energy_storages']['s'] = dict()
                processed_results['scenarios'][s_m][s_o]['energy_storages']['soc'] = dict()
                processed_results['scenarios'][s_m][s_o]['energy_storages']['soc_percent'] = dict()

            if params.relaxed_model or params.ess_relax or params.fl_relax:
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks'] = dict()
                if params.ess_relax:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['ch'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['dch'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['comp'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['day_balance'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['ch'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['dch'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['comp'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['day_balance'] = dict()
                if params.relaxed_model:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['nodes'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['nodes']['gen_vg'] = dict()
                if params.fl_relax:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['nodes']['day_balance'] = dict()

            # Voltage
            for i in model.nodes:
                node_id = network.nodes[i].bus_i
                processed_results['scenarios'][s_m][s_o]['voltage']['vmag'][node_id] = []
                processed_results['scenarios'][s_m][s_o]['voltage']['vang'][node_id] = []
                for p in model.periods:
                    e = pe.value(model.e_actual[i, s_m, s_o, p])
                    f = pe.value(model.f_actual[i, s_m, s_o, p])
                    v_mag = sqrt(pe.value(model.vmag_sqr[i, s_m, s_o, p]))
                    v_ang = atan2(f, e) * (180.0 / pi)
                    processed_results['scenarios'][s_m][s_o]['voltage']['vmag'][node_id].append(v_mag)
                    processed_results['scenarios'][s_m][s_o]['voltage']['vang'][node_id].append(v_ang)

            # Consumption
            for i in model.nodes:
                node = network.nodes[i]
                processed_results['scenarios'][s_m][s_o]['consumption']['pc'][node.bus_i] = []
                processed_results['scenarios'][s_m][s_o]['consumption']['qc'][node.bus_i] = []
                processed_results['scenarios'][s_m][s_o]['consumption']['pc_net'][node.bus_i] = [0.00 for _ in range(network.num_instants)]
                processed_results['scenarios'][s_m][s_o]['consumption']['qc_net'][node.bus_i] = [0.00 for _ in range(network.num_instants)]
                if params.fl_reg:
                    processed_results['scenarios'][s_m][s_o]['consumption']['p_up'][node.bus_i] = []
                    processed_results['scenarios'][s_m][s_o]['consumption']['p_down'][node.bus_i] = []
                if params.l_curt:
                    processed_results['scenarios'][s_m][s_o]['consumption']['pc_curt'][node.bus_i] = []
                    processed_results['scenarios'][s_m][s_o]['consumption']['qc_curt'][node.bus_i] = []
                for p in model.periods:
                    pc = pe.value(model.pc[i, s_m, s_o, p]) * network.baseMVA
                    qc = pe.value(model.qc[i, s_m, s_o, p]) * network.baseMVA
                    processed_results['scenarios'][s_m][s_o]['consumption']['pc'][node.bus_i].append(pc)
                    processed_results['scenarios'][s_m][s_o]['consumption']['qc'][node.bus_i].append(qc)
                    processed_results['scenarios'][s_m][s_o]['consumption']['pc_net'][node.bus_i][p] += pc
                    processed_results['scenarios'][s_m][s_o]['consumption']['qc_net'][node.bus_i][p] += qc
                    if params.fl_reg:
                        pup = pe.value(model.flex_p_up[i, s_m, s_o, p]) * network.baseMVA
                        pdown = pe.value(model.flex_p_down[i, s_m, s_o, p]) * network.baseMVA
                        processed_results['scenarios'][s_m][s_o]['consumption']['p_up'][node.bus_i].append(pup)
                        processed_results['scenarios'][s_m][s_o]['consumption']['p_down'][node.bus_i].append(pdown)
                        processed_results['scenarios'][s_m][s_o]['consumption']['pc_net'][node.bus_i][p] += pup - pdown
                    if params.l_curt:
                        pc_curt = pe.value(model.pc_curt[i, s_m, s_o, p]) * network.baseMVA
                        qc_curt = pe.value(model.qc_curt[i, s_m, s_o, p]) * network.baseMVA
                        processed_results['scenarios'][s_m][s_o]['consumption']['pc_curt'][node.bus_i].append(pc_curt)
                        processed_results['scenarios'][s_m][s_o]['consumption']['pc_net'][node.bus_i][p] -= pc_curt
                        processed_results['scenarios'][s_m][s_o]['consumption']['qc_curt'][node.bus_i].append(qc_curt)
                        processed_results['scenarios'][s_m][s_o]['consumption']['qc_net'][node.bus_i][p] -= qc_curt

            # Generation
            for g in model.generators:
                processed_results['scenarios'][s_m][s_o]['generation']['pg'][g] = []
                processed_results['scenarios'][s_m][s_o]['generation']['qg'][g] = []
                processed_results['scenarios'][s_m][s_o]['generation']['pg_net'][g] = [0.00 for _ in range(network.num_instants)]
                if params.rg_curt:
                    processed_results['scenarios'][s_m][s_o]['generation']['pg_curt'][g] = []
                for p in model.periods:
                    pg = pe.value(model.pg[g, s_m, s_o, p]) * network.baseMVA
                    qg = pe.value(model.qg[g, s_m, s_o, p]) * network.baseMVA
                    processed_results['scenarios'][s_m][s_o]['generation']['pg'][g].append(pg)
                    processed_results['scenarios'][s_m][s_o]['generation']['qg'][g].append(qg)
                    processed_results['scenarios'][s_m][s_o]['generation']['pg_net'][g][p] += pg
                    if params.rg_curt:
                        pg_curt = pe.value(model.pg_curt[g, s_m, s_o, p]) * network.baseMVA
                        processed_results['scenarios'][s_m][s_o]['generation']['pg_curt'][g].append(pg_curt)
                        processed_results['scenarios'][s_m][s_o]['generation']['pg_net'][g][p] -= pg_curt

            # Branch current, transformers' ratio
            for k in model.branches:

                rating = network.branches[k].rate / network.baseMVA
                if rating == 0.0:
                    rating = BRANCH_UNKNOWN_RATING

                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['pij'][k] = []
                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['pji'][k] = []
                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['qij'][k] = []
                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['qji'][k] = []
                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['sij'][k] = []
                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['sji'][k] = []
                processed_results['scenarios'][s_m][s_o]['branches']['current_perc'][k] = []
                processed_results['scenarios'][s_m][s_o]['branches']['losses'][k] = []
                if network.branches[k].is_transformer:
                    processed_results['scenarios'][s_m][s_o]['branches']['ratio'][k] = []
                for p in model.periods:

                    # Power flows
                    pij, qij = _get_branch_power_flow(network, params, network.branches[k], network.branches[k].fbus, network.branches[k].tbus, model, s_m, s_o, p)
                    pji, qji = _get_branch_power_flow(network, params, network.branches[k], network.branches[k].tbus, network.branches[k].fbus, model, s_m, s_o, p)
                    sij_sqr = pij**2 + qij**2
                    sji_sqr = pji**2 + qji**2
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['pij'][k].append(pij)
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['pji'][k].append(pji)
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['qij'][k].append(qij)
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['qji'][k].append(qji)
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['sij'][k].append(sqrt(sij_sqr))
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['sji'][k].append(sqrt(sji_sqr))

                    # Current
                    iij_sqr = abs(pe.value(model.iij_sqr[k, s_m, s_o, p]))
                    processed_results['scenarios'][s_m][s_o]['branches']['current_perc'][k].append(sqrt(iij_sqr) / rating)

                    # Losses (active power)
                    p_losses = _get_branch_power_losses(network, params, model, k, s_m, s_o, p)
                    processed_results['scenarios'][s_m][s_o]['branches']['losses'][k].append(p_losses)

                    # Ratio
                    if network.branches[k].is_transformer:
                        r_ij = pe.value(model.r[k, s_m, s_o, p])
                        processed_results['scenarios'][s_m][s_o]['branches']['ratio'][k].append(r_ij)

            # Energy Storage devices
            if params.es_reg:
                for e in model.energy_storages:
                    node_id = network.energy_storages[e].bus
                    capacity = network.energy_storages[e].e * network.baseMVA
                    processed_results['scenarios'][s_m][s_o]['energy_storages']['p'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['energy_storages']['q'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['energy_storages']['s'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['energy_storages']['soc'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['energy_storages']['soc_percent'][node_id] = []
                    for p in model.periods:
                        if capacity > 0.0:
                            sch = pe.value(model.es_sch[e, s_m, s_o, p]) * network.baseMVA
                            pch = pe.value(model.es_pch[e, s_m, s_o, p]) * network.baseMVA
                            qch = pe.value(model.es_qch[e, s_m, s_o, p]) * network.baseMVA
                            sdch = pe.value(model.es_sdch[e, s_m, s_o, p]) * network.baseMVA
                            pdch = pe.value(model.es_pdch[e, s_m, s_o, p]) * network.baseMVA
                            qdch = pe.value(model.es_qdch[e, s_m, s_o, p]) * network.baseMVA
                            soc_ess = pe.value(model.es_soc[e, s_m, s_o, p]) * network.baseMVA
                            processed_results['scenarios'][s_m][s_o]['energy_storages']['p'][node_id].append(pch - pdch)
                            processed_results['scenarios'][s_m][s_o]['energy_storages']['q'][node_id].append(qch - qdch)
                            processed_results['scenarios'][s_m][s_o]['energy_storages']['s'][node_id].append(sch - sdch)
                            processed_results['scenarios'][s_m][s_o]['energy_storages']['soc'][node_id].append(soc_ess)
                            processed_results['scenarios'][s_m][s_o]['energy_storages']['soc_percent'][node_id].append(soc_ess / capacity)
                        else:
                            processed_results['scenarios'][s_m][s_o]['energy_storages']['p'][node_id].append('N/A')
                            processed_results['scenarios'][s_m][s_o]['energy_storages']['q'][node_id].append('N/A')
                            processed_results['scenarios'][s_m][s_o]['energy_storages']['s'][node_id].append('N/A')
                            processed_results['scenarios'][s_m][s_o]['energy_storages']['soc'][node_id].append('N/A')
                            processed_results['scenarios'][s_m][s_o]['energy_storages']['soc_percent'][node_id].append('N/A')

            # Flexible loads
            if params.fl_reg:
                for i in model.nodes:
                    node_id = network.nodes[i].bus_i
                    processed_results['scenarios'][s_m][s_o]['consumption']['p_up'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['consumption']['p_down'][node_id] = []
                    for p in model.periods:
                        p_up = pe.value(model.flex_p_up[i, s_m, s_o, p]) * network.baseMVA
                        p_down = pe.value(model.flex_p_down[i, s_m, s_o, p]) * network.baseMVA
                        processed_results['scenarios'][s_m][s_o]['consumption']['p_up'][node_id].append(p_up)
                        processed_results['scenarios'][s_m][s_o]['consumption']['p_down'][node_id].append(p_down)

            # Shared Energy Storages
            for e in model.shared_energy_storages:
                node_id = network.shared_energy_storages[e].bus
                capacity = network.shared_energy_storages[e].e * network.baseMVA
                processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['p'][node_id] = []
                processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['q'][node_id] = []
                processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['s'][node_id] = []
                processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['soc'][node_id] = []
                processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['soc_percent'][node_id] = []
                for p in model.periods:
                    if not isclose(capacity, 0.0, abs_tol=1e-3):
                        p_ess = pe.value(model.shared_es_pch[e, s_m, s_o, p] - model.shared_es_pdch[e, s_m, s_o, p]) * network.baseMVA
                        q_ess = pe.value(model.shared_es_qch[e, s_m, s_o, p] - model.shared_es_qdch[e, s_m, s_o, p]) * network.baseMVA
                        s_ess = pe.value(model.shared_es_sch[e, s_m, s_o, p] - model.shared_es_sdch[e, s_m, s_o, p]) * network.baseMVA
                        soc_ess = pe.value(model.shared_es_soc[e, s_m, s_o, p]) * network.baseMVA
                        processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['p'][node_id].append(p_ess)
                        processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['q'][node_id].append(q_ess)
                        processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['s'][node_id].append(s_ess)
                        processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['soc'][node_id].append(soc_ess)
                        processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['soc_percent'][node_id].append(soc_ess / capacity)
                    else:
                        processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['p'][node_id].append('N/A')
                        processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['q'][node_id].append('N/A')
                        processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['s'][node_id].append('N/A')
                        processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['soc'][node_id].append('N/A')
                        processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['soc_percent'][node_id].append('N/A')

            # Slack variable penalties
            if params.relaxed_model or params.ess_relax or params.fl_relax:

                # Shared ESS
                if params.relaxed_model:
                    for e in model.shared_energy_storages:
                        node_id = network.shared_energy_storages[e].bus
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['ch'][node_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['dch'][node_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc'][node_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['comp'][node_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['day_balance'][node_id] = pe.value(model.penalty_shared_es_soc_day_balance[e, s_m, s_o])
                        for p in model.periods:
                            slack_ch = pe.value(model.penalty_shared_es_ch[e, s_m, s_o, p])
                            slack_dch = pe.value(model.penalty_shared_es_ch[e, s_m, s_o, p])
                            slack_soc = pe.value(model.penalty_shared_es_soc[e, s_m, s_o, p])
                            slack_comp = pe.value(model.penalty_shared_es_comp[e, s_m, s_o, p])
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['ch'][node_id].append(slack_ch)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['dch'][node_id].append(slack_dch)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc'][node_id].append(slack_soc)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['comp'][node_id].append(slack_comp)

                # ESS
                if params.es_reg and params.ess_relax:
                    for e in model.energy_storages:
                        node_id = network.energy_storages[e].bus
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['ch'][node_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['dch'][node_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc'][node_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['comp'][node_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['day_balance'][node_id] = pe.value(model.penalty_shared_es_soc_day_balance[e, s_m, s_o])
                        for p in model.periods:
                            slack_ch = pe.value(model.penalty_es_ch[e, s_m, s_o, p])
                            slack_dch = pe.value(model.penalty_es_ch[e, s_m, s_o, p])
                            slack_soc = pe.value(model.penalty_es_soc[e, s_m, s_o, p])
                            slack_comp = pe.value(model.penalty_es_comp[e, s_m, s_o, p])
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['ch'][node_id].append(slack_ch)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['dch'][node_id].append(slack_dch)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc'][node_id].append(slack_soc)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['comp'][node_id].append(slack_comp)

                # PV bus voltage set-point
                if params.relaxed_model:
                    for i in model.nodes:
                        node_id = network.nodes[i].bus_i
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['nodes']['gen_vg'][node_id] = []
                        for p in model.periods:
                            slack_vg = pe.value(model.penalty_gen_vg[i, s_m, s_o, p])
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['nodes']['gen_vg'][node_id].append(slack_vg)

                # Flex daily balance
                if params.fl_reg and params.fl_relax:
                    for i in model.nodes:
                        node_id = network.nodes[i].bus_i
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['nodes']['day_balance'][node_id] = pe.value(model.penalty_flex_day_balance[e, s_m, s_o])
                        for p in model.periods:
                            slack_vg = pe.value(model.penalty_gen_vg[i, s_m, s_o, p])
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['nodes']['gen_vg'][node_id].append(slack_vg)

    return processed_results


def _process_results_interface_power_flow(network, model):

    results = dict()

    if network.is_transmission:
        for dn in model.active_distribution_networks:

            node_id = network.active_distribution_network_nodes[dn]
            node_idx = network.get_node_idx(node_id)

            # Power flow results per market and operation scenario
            results[node_id] = dict()
            for s_m in model.scenarios_market:
                results[node_id][s_m] = dict()
                for s_o in model.scenarios_operation:
                    results[node_id][s_m][s_o] = dict()
                    results[node_id][s_m][s_o]['p'] = [0.0 for _ in model.periods]
                    results[node_id][s_m][s_o]['q'] = [0.0 for _ in model.periods]
                    for p in model.periods:
                        results[node_id][s_m][s_o]['p'][p] = pe.value(model.pc[node_idx, s_m, s_o, p]) * network.baseMVA
                        results[node_id][s_m][s_o]['q'][p] = pe.value(model.qc[node_idx, s_m, s_o, p]) * network.baseMVA
    else:

        # Power flow results per market and operation scenario
        ref_gen_idx = network.get_reference_gen_idx()
        for s_m in model.scenarios_market:
            results[s_m] = dict()
            for s_o in model.scenarios_operation:
                results[s_m][s_o] = dict()
                results[s_m][s_o]['p'] = [0.0 for _ in model.periods]
                results[s_m][s_o]['q'] = [0.0 for _ in model.periods]
                for p in model.periods:
                    results[s_m][s_o]['p'][p] = pe.value(model.pg[ref_gen_idx, s_m, s_o, p]) * network.baseMVA
                    results[s_m][s_o]['q'][p] = pe.value(model.qg[ref_gen_idx, s_m, s_o, p]) * network.baseMVA

    return results


def _compute_objective_function_value(network, model, params):

    obj = 0.0

    if params.obj_type == OBJ_MIN_COST:

        c_p = network.cost_energy_p
        #c_q = network.cost_energy_q

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:

                obj_scenario = 0.0

                # Generation -- paid at market price
                for g in model.generators:
                    if network.generators[g].is_controllable():
                        for p in model.periods:
                            obj_scenario += c_p[s_m][p] * network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])

                # Demand side flexibility
                if params.fl_reg:
                    for i in model.nodes:
                        node = network.nodes[i]
                        for p in model.periods:
                            cost_flex = node.flexibility.cost[p]
                            flex_down = pe.value(model.flex_p_down[i, s_m, s_o, p])
                            obj_scenario += cost_flex * network.baseMVA * flex_down

                # Load curtailment
                if params.l_curt:
                    for i in model.nodes:
                        for p in model.periods:
                            pc_curt = pe.value(model.pc_curt[i, s_m, s_o, p])
                            qc_curt = pe.value(model.qc_curt[i, s_m, s_o, p])
                            obj_scenario += (COST_CONSUMPTION_CURTAILMENT * network.baseMVA) * pc_curt
                            obj_scenario += (COST_CONSUMPTION_CURTAILMENT * network.baseMVA) * qc_curt

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = pe.value(model.pg_curt[g, s_m, s_o, p])
                            obj_scenario += (COST_GENERATION_CURTAILMENT * network.baseMVA) * pg_curt

                obj += obj_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:

                obj_scenario = 0.0

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = model.pg_curt[g, s_m, s_o, p]
                            obj_scenario += PENALTY_GENERATION_CURTAILMENT * pg_curt

                # Consumption curtailment
                if params.l_curt:
                    for i in model.nodes:
                        for p in model.periods:
                            pc_curt = model.pc_curt[i, s_m, s_o, p]
                            obj_scenario += PENALTY_LOAD_CURTAILMENT * pc_curt

    return obj


def _compute_generation_cost(network, model):

    gen_cost = 0.0

    c_p = network.cost_energy_p
    #c_q = network.cost_energy_q

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            gen_cost_scenario = 0.0
            for g in model.generators:
                if network.generators[g].is_controllable():
                    for p in model.periods:
                        gen_cost_scenario += c_p[s_m][p] * network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])
                        #gen_cost_scenario += c_q[s_m][p] * network.baseMVA * pe.value(model.qg[g, s_m, s_o, p])

            gen_cost += gen_cost_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return gen_cost


def _compute_total_load(network, model, params):

    total_load = 0.0

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_load_scenario = 0.0
            for i in model.nodes:
                for p in model.periods:
                    total_load_scenario += network.baseMVA * pe.value(model.pc[i, s_m, s_o, p])
                    if params.l_curt:
                        total_load_scenario -= network.baseMVA * pe.value(model.pc_curt[i, s_m, s_o, p])

            total_load += total_load_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return total_load


def _compute_total_generation(network, model, params):

    total_gen = 0.0

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_gen_scenario = 0.0
            for g in model.generators:
                for p in model.periods:
                    total_gen_scenario += network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])
                    if params.rg_curt:
                        total_gen_scenario -= network.baseMVA * pe.value(model.pg_curt[g, s_m, s_o, p])

            total_gen += total_gen_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return total_gen


def _compute_conventional_generation(network, model, params):

    total_gen = 0.0

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_gen_scenario = 0.0
            for g in model.generators:
                if network.generators[g].gen_type == GEN_CONV:
                    for p in model.periods:
                        total_gen_scenario += network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])
                        if params.rg_curt:
                            total_gen_scenario -= network.baseMVA * pe.value(model.pg_curt[g, s_m, s_o, p])

            total_gen += total_gen_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return total_gen


def _compute_renewable_generation(network, model, params):

    total_renewable_gen = 0.0

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_renewable_gen_scenario = 0.0
            for g in model.generators:
                if network.generators[g].is_renewable():
                    for p in model.periods:
                        total_renewable_gen_scenario += network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])
                        if params.rg_curt:
                            total_renewable_gen_scenario -= network.baseMVA * pe.value(model.pg_curt[g, s_m, s_o, p])

            total_renewable_gen += total_renewable_gen_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return total_renewable_gen


def _compute_losses(network, model, params):

    power_losses = 0.0

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            power_losses_scenario = 0.0
            for k in model.branches:
                for p in model.periods:
                    power_losses_scenario += _get_branch_power_losses(network, params, model, k, s_m, s_o, p)

            power_losses += power_losses_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return power_losses


def _compute_generation_curtailment(network, model, params):

    gen_curtailment = 0.0

    if params.rg_curt:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                gen_curtailment_scenario = 0.0
                for g in model.generators:
                    if network.generators[g].is_curtaillable():
                        for p in model.periods:
                            gen_curtailment_scenario += pe.value(model.pg_curt[g, s_m, s_o, p]) * network.baseMVA

                gen_curtailment += gen_curtailment_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return gen_curtailment


def _compute_load_curtailment(network, model, params):

    load_curtailment = 0.0

    if params.l_curt:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                load_curtailment_scenario = 0.0
                for i in model.nodes:
                    for p in model.periods:
                        load_curtailment_scenario += pe.value(model.pc_curt[i, s_m, s_o, p]) * network.baseMVA

                load_curtailment += load_curtailment_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return load_curtailment


def _compute_flexibility_used(network, model, params):

    flexibility_used = 0.0

    if params.fl_reg:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                flexibility_used_scenario = 0.0
                for i in model.nodes:
                    for p in model.periods:
                        flexibility_used_scenario += pe.value(model.flex_p_up[i, s_m, s_o, p]) * network.baseMVA

                flexibility_used += flexibility_used_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return flexibility_used


def _get_branch_power_losses(network, params, model, branch_idx, s_m, s_o, p):

    # Active power flow, from i to j and from j to i
    branch = network.branches[branch_idx]
    pij, _ = _get_branch_power_flow(network, params, branch, branch.fbus, branch.tbus, model, s_m, s_o, p)
    pji, _ = _get_branch_power_flow(network, params, branch, branch.tbus, branch.fbus, model, s_m, s_o, p)

    return abs(pij + pji)


def _get_branch_power_flow(network, params, branch, fbus, tbus, model, s_m, s_o, p):

    fbus_idx = network.get_node_idx(fbus)
    tbus_idx = network.get_node_idx(tbus)
    branch_idx = network.get_branch_idx(branch)
    if branch.fbus == fbus:
        direction = 1
    else:
        direction = 0

    rij = 1 / pe.value(model.r[branch_idx, s_m, s_o, p])
    ei = pe.value(model.e_actual[fbus_idx, s_m, s_o, p])
    fi = pe.value(model.f_actual[fbus_idx, s_m, s_o, p])
    ej = pe.value(model.e_actual[tbus_idx, s_m, s_o, p])
    fj = pe.value(model.f_actual[tbus_idx, s_m, s_o, p])

    if direction:
        pij = branch.g * (ei ** 2 + fi ** 2) * rij**2
        pij -= branch.g * (ei * ej + fi * fj) * rij
        pij -= branch.b * (fi * ej - ei * fj) * rij

        qij = - (branch.b + branch.b_sh * 0.50) * (ei ** 2 + fi ** 2) * rij**2
        qij += branch.b * (ei * ej + fi * fj) * rij
        qij -= branch.g * (fi * ej - ei * fj) * rij
    else:
        pij = branch.g * (ei ** 2 + fi ** 2)
        pij -= branch.g * (ei * ej + fi * fj) * rij
        pij -= branch.b * (fi * ej - ei * fj) * rij

        qij = - (branch.b + branch.b_sh * 0.50) * (ei ** 2 + fi ** 2)
        qij += branch.b * (ei * ej + fi * fj) * rij
        qij -= branch.g * (fi * ej - ei * fj) * rij

    return pij * network.baseMVA, qij * network.baseMVA


# ======================================================================================================================
#   Other (aux) functions
# ======================================================================================================================
def _perform_network_check(network):

    n_bus = len(network.nodes)
    if n_bus == 0:
        print(f'[ERROR] Reading network {network.name}. No nodes imported.')
        exit(ERROR_NETWORK_FILE)

    n_branch = len(network.branches)
    if n_branch == 0:
        print(f'[ERROR] Reading network {network.name}. No branches imported.')
        exit(ERROR_NETWORK_FILE)


def _pre_process_network(network):

    processed_nodes = []
    for node in network.nodes:
        if node.type != BUS_ISOLATED:
            processed_nodes.append(node)

    processed_gens = []
    for gen in network.generators:
        node_type = network.get_node_type(gen.bus)
        if node_type != BUS_ISOLATED:
            processed_gens.append(gen)

    processed_branches = []
    for branch in network.branches:

        if not branch.is_connected():  # If branch is disconnected for all days and periods, remove
            continue

        if branch.pre_processed:
            continue

        fbus, tbus = branch.fbus, branch.tbus
        fnode_type = network.get_node_type(fbus)
        tnode_type = network.get_node_type(tbus)
        if fnode_type == BUS_ISOLATED or tnode_type == BUS_ISOLATED:
            branch.pre_processed = True
            continue

        parallel_branches = [branch for branch in network.branches if ((branch.fbus == fbus and branch.tbus == tbus) or (branch.fbus == tbus and branch.tbus == fbus))]
        connected_parallel_branches = [branch for branch in parallel_branches if branch.is_connected()]
        if len(connected_parallel_branches) > 1:
            processed_branch = connected_parallel_branches[0]
            r_eq, x_eq, g_eq, b_eq = _pre_process_parallel_branches(connected_parallel_branches)
            processed_branch.r = r_eq
            processed_branch.x = x_eq
            processed_branch.g_sh = g_eq
            processed_branch.b_sh = b_eq
            processed_branch.rate = sum([branch.rate for branch in connected_parallel_branches])
            processed_branch.ratio = branch.ratio
            processed_branch.pre_processed = True
            for branch in parallel_branches:
                branch.pre_processed = True
            processed_branches.append(processed_branch)
        else:
            for branch in parallel_branches:
                branch.pre_processed = True
            for branch in connected_parallel_branches:
                processed_branches.append(branch)

    network.nodes = processed_nodes
    network.generators = processed_gens
    network.branches = processed_branches
    for branch in network.branches:
        branch.pre_processed = False


def _pre_process_parallel_branches(branches):
    branch_impedances = [complex(branch.r, branch.x) for branch in branches]
    branch_shunt_admittance = [complex(branch.g_sh, branch.b_sh) for branch in branches]
    z_eq = 1/sum([(1/impedance) for impedance in branch_impedances])
    ysh_eq = sum([admittance for admittance in branch_shunt_admittance])
    return abs(z_eq.real), abs(z_eq.imag), ysh_eq.real, ysh_eq.imag


def _get_info_from_results(results, info_string):
    i = str(results).lower().find(info_string.lower()) + len(info_string)
    value = ''
    while str(results)[i] != '\n':
        value = value + str(results)[i]
        i += 1
    return value
