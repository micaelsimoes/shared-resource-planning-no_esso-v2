from solver_parameters import SolverParameters
from helper_functions import *

# ======================================================================================================================
#  Energy Storage Parameters
# ======================================================================================================================
class SharedEnergyStorageParameters:

    def __init__(self):
        self.budget = 1e6                           # 1 M m.u.
        self.max_capacity = 2.50                    # Max energy capacity (related to space constraints)
        self.min_pe_factor = 0.10                   # Minimum S/E factor (related to the ESS technology)
        self.max_pe_factor = 4.00                   # Maximum S/E factor (related to the ESS technology)
        self.ess_relax = True                       # Charging/Discharging modeling method (True: McCormick envelopes, False: NLP model)
        self.plot_results = False                   # Plot results
        self.print_results_to_file = False          # Write results to file
        self.verbose = False                        # Verbose -- Bool
        self.solver_params = SolverParameters()     # Solver Parameters

    def read_parameters_from_file(self, filename):
        _read_parameters_from_file(self, filename)


def _read_parameters_from_file(planning_parameters, filename):

    params_data = convert_json_to_dict(read_json_file(filename))

    planning_parameters.budget = float(params_data['budget'])
    planning_parameters.max_capacity = float(params_data['max_capacity'])
    planning_parameters.min_pe_factor = float(params_data['min_pe_factor'])
    planning_parameters.max_pe_factor = float(params_data['max_pe_factor'])
    planning_parameters.ess_relax = bool(params_data['ess_relax'])
    planning_parameters.plot_results = bool(params_data['plot_results'])
    planning_parameters.print_results_to_file = bool(params_data['print_results_to_file'])
    planning_parameters.solver_params.read_solver_parameters(params_data['solver'])
