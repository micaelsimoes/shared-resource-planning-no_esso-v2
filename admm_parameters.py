# ======================================================================================================================
#  Class ADMM Parameters
# ======================================================================================================================
class ADMMParameters:

    def __init__(self):
        self.tol = 1e-3
        self.num_max_iters = 1000
        self.rho = {'pf': dict(), 'ess': dict()}

    def read_parameters_from_file(self, params_data):
        _read_parameters_from_file(self, params_data)


def _read_parameters_from_file(admm_params, params_data):
    admm_params.tol = float(params_data['tol'])
    admm_params.num_max_iters = int(params_data['num_max_iters'])
    admm_params.adaptive_penalty = bool(params_data['adaptive_penalty'])
    admm_params.rho['pf'] = params_data['rho']['pf']
    admm_params.rho['ess'] = params_data['rho']['ess']
