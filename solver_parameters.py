import os


# ============================================================================================
#   Class SolverParameters
# ============================================================================================
class SolverParameters:

    def __init__(self):
        self.solver = 'ipopt'
        self.linear_solver = 'ma57'
        self.nlp_solver = 'ipopt'
        self.solver_path = os.environ['IPOPTDIR']
        self.solver_tol = 1e-6
        self.verbose = False

    def read_solver_parameters(self, solver_data):
        _read_solver_parameters(self, solver_data)


def _read_solver_parameters(parameters, solver_data):
    parameters.solver = solver_data['name']
    parameters.linear_solver = solver_data['linear_solver']
    parameters.solver_tol = solver_data['solver_tol']
    parameters.verbose = solver_data['verbose']
