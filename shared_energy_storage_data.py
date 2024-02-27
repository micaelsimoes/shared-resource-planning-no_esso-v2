import os
import pandas as pd
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

    def update_data_with_candidate_solution(self, candidate_solution):
        for year in self.years:
            for shared_ess in self.shared_energy_storages[year]:
                shared_ess.s = candidate_solution[shared_ess.bus][year]['s']
                shared_ess.e = candidate_solution[shared_ess.bus][year]['e']
                shared_ess.e_init = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_RELATIVE_INIT_SOC
                shared_ess.e_min = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_MIN_ENERGY_STORED
                shared_ess.e_max = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_MAX_ENERGY_STORED


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
