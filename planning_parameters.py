from admm_parameters import ADMMParameters
from benders_parameters import BendersParameters
from helper_functions import *


# ======================================================================================================================
#  Class Planning Parameters
# ======================================================================================================================
class PlanningParameters:

    def __init__(self):
        self.admm = ADMMParameters()
        self.benders = BendersParameters()

    def read_parameters_from_file(self, filename):
        params_data = convert_json_to_dict(read_json_file(filename))
        self.benders.read_parameters_from_file(params_data['benders'])
        self.admm.read_parameters_from_file(params_data['admm'])
