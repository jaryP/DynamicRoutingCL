from avalanche.training import DER, Replay, GEM, Cumulative, Naive, \
    JointTraining, ICaRL

from .routing import ContinuosRouting

from .ssil import SeparatedSoftmax

# AVAILABLE_METHODS = {
#     'ssil': SeparatedSoftmax,
#     'der': DER,
#     'replay': Replay,
#     'gem': GEM,
#     'cumulative': Cumulative,
#     'naive': Naive,
#     'joint': JointTraining,
#     'icarl': ICaRL
# }