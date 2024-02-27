# ======================================================================================================================
#  Class Benders' Parameters
# ======================================================================================================================
class BendersParameters:

    def __init__(self):
        self.tol_abs = 1e3
        self.tol_rel = 1e-2
        self.num_max_iters = 1000

    def read_parameters_from_file(self, params_data):
        _read_parameters_from_file(self, params_data)


def _read_parameters_from_file(benders_params, params_data):
    benders_params.tol_abs = float(params_data['tol_abs'])
    benders_params.tol_rel = float(params_data['tol_rel'])
    benders_params.num_max_iters = int(params_data['num_max_iters'])
