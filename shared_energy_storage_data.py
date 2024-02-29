import os
import pandas as pd
import pyomo.opt as po
import pyomo.environ as pe
from shared_energy_storage import SharedEnergyStorage
from shared_energy_storage_parameters import SharedEnergyStorageParameters
from helper_functions import *


# ======================================================================================================================
#  SHARED ENERGY STORAGE Information
# ======================================================================================================================
class SharedEnergyStorageData:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.data_file = str()
        self.params_file = str()
        self.shared_energy_storages = dict()
        self.cost_investment = dict()
        self.params = SharedEnergyStorageParameters()

    def create_shared_energy_storages(self, planning_problem):
        for year in planning_problem.years:
            self.shared_energy_storages[year] = list()
            for node_id in planning_problem.transmission_network.active_distribution_network_nodes:
                shared_energy_storage = SharedEnergyStorage()
                shared_energy_storage.bus = node_id
                shared_energy_storage.dn_name = planning_problem.distribution_networks[node_id].name
                self.shared_energy_storages[year].append(shared_energy_storage)

    def read_shared_energy_storage_data_from_file(self):
        filename = os.path.join(self.data_dir, 'Shared ESS', self.data_file)
        _read_shared_energy_storage_data_from_file(self, filename)

    def read_parameters_from_file(self):
        filename = os.path.join(self.data_dir, 'Shared ESS', self.params_file)
        self.params.read_parameters_from_file(filename)

    def optimize(self, model, from_warm_start=False):
        return _optimize(model, self.params.solver_params, from_warm_start=from_warm_start)

    def get_candidate_solution(self, model):
        years = [year for year in self.years]
        candidate_solution = {'investment': {}, 'total_capacity': {}}
        for e in model.energy_storages:
            node_id = self.shared_energy_storages[years[0]][e].bus
            candidate_solution['investment'][node_id] = dict()
            candidate_solution['total_capacity'][node_id] = dict()
            for y in model.years:
                year = years[y]
                candidate_solution['investment'][node_id][year] = dict()
                candidate_solution['investment'][node_id][year]['s'] = abs(pe.value(model.es_s_invesment[e, y]))
                candidate_solution['investment'][node_id][year]['e'] = abs(pe.value(model.es_e_invesment[e, y]))
                candidate_solution['total_capacity'][node_id][year] = dict()
                candidate_solution['total_capacity'][node_id][year]['s'] = abs(pe.value(model.es_s_rated[e, y]))
                candidate_solution['total_capacity'][node_id][year]['e'] = abs(pe.value(model.es_e_rated[e, y]))
        return candidate_solution

    def update_data_with_candidate_solution(self, candidate_solution):
        for year in self.years:
            for shared_ess in self.shared_energy_storages[year]:
                shared_ess.s = candidate_solution[shared_ess.bus][year]['s']
                shared_ess.e = candidate_solution[shared_ess.bus][year]['e']
                shared_ess.e_init = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_RELATIVE_INIT_SOC
                shared_ess.e_min = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_MIN_ENERGY_STORED
                shared_ess.e_max = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_MAX_ENERGY_STORED


# ======================================================================================================================
#  OPTIMIZATION  functions
# ======================================================================================================================
def _optimize(model, params, from_warm_start=False):

    solver = po.SolverFactory(params.solver, executable=params.solver_path, tee=params.verbose)

    if from_warm_start:
        model.ipopt_zL_in.update(model.ipopt_zL_out)
        model.ipopt_zU_in.update(model.ipopt_zU_out)
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['warm_start_bound_push'] = 1e-9
        solver.options['warm_start_mult_bound_push'] = 1e-9
        solver.options['mu_init'] = 1e-9

    if params.verbose:
        solver.options['print_level'] = 6
        solver.options['output_file'] = 'optim_log.txt'

    if params.solver == 'ipopt':
        solver.options['tol'] = params.solver_tol
        solver.options['acceptable_tol'] = params.solver_tol * 1e3
        solver.options['acceptable_iter'] = 5
        solver.options['nlp_scaling_method'] = 'none'
        solver.options['max_iter'] = 10000
        solver.options['linear_solver'] = params.linear_solver

    result = solver.solve(model, tee=params.verbose)
    '''
    if not result.solver.status == po.SolverStatus.ok:
        import logging
        from pyomo.util.infeasible import log_infeasible_constraints
        filename = os.path.join(os.getcwd(), 'master_problem.log')
        print(log_infeasible_constraints(model, log_expression=True, log_variables=True))
        logging.basicConfig(filename=filename, encoding='utf-8', level=logging.INFO)
    '''

    return result


# ======================================================================================================================
#  SHARED ESS DATA read functions
# ======================================================================================================================
def _read_shared_energy_storage_data_from_file(shared_ess_data, filename):
    try:
        investment_costs = _get_investment_costs_from_excel_file(filename, 'Investment Cost', len(shared_ess_data.years))
        shared_ess_data.cost_investment = investment_costs
    except:
        print(f'[ERROR] File {filename}. Exiting...')
        exit(ERROR_OPERATIONAL_DATA_FILE)


def _get_investment_costs_from_excel_file(filename, sheet_name, num_years):

    try:

        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        data = {
            'power_capacity': dict(),
            'energy_capacity': dict()
        }

        for i in range(num_years):

            year = int(df.iloc[0, i + 1])

            if is_number(df.iloc[1, i + 1]):
                data['power_capacity'][year] = float(df.iloc[1, i + 1])

            if is_number(df.iloc[2, i + 1]):
                data['energy_capacity'][year] = float(df.iloc[2, i + 1])

        return data

    except:
        print('[ERROR] Workbook {}. Sheet {} does not exist.'.format(filename, sheet_name))
        exit(ERROR_MARKET_DATA_FILE)


