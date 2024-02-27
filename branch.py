# ============================================================================================
#   Class Branch
# ============================================================================================
class Branch:

    def __init__(self):
        self.branch_id = 0           # Branch ID
        self.fbus = 0                # f, from bus number
        self.tbus = 0                # t, to bus number
        self.r = 0.0                 # r, resistance (p.u.)
        self.x = 0.0                 # x, reactance (p.u.)
        self.g_sh = 0.0              # g, total line charging conductance (p.u.)
        self.b_sh = 0.0              # b, total line charging susceptance (p.u.)
        self.rate = 0.0              # rateA, MVA rating A (long term rating)
        self.ratio = 0.0             # ratio, transformer off nominal turns ratio ( = 0 for lines )
                                     #  (taps at 'from' bus, impedance at 'to' bus,
                                     #   i.e. if r = x = 0, then ratio = Vf / Vt)
        self.status = True           # initial branch status
        self.pre_processed = False
        self.is_transformer = False  # Indicates if the branch is a transformer
        self.vmag_reg = False        # Indicates if transformer has voltage magnitude regulation

    def is_connected(self):
        return self.status
