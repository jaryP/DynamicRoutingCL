from avalanche.training import DER, Replay, GEM, Cumulative, Naive, \
    JointTraining, ICaRL

from .routing import ContinuosRouting

from methods.strategies import SeparatedSoftmaxIncrementalLearning

# AVAILABLE_METHODS = {
#     'ssil': SeparatedSoftmaxIncrementalLearning,
#     'der': DER,
#     'replay': Replay,
#     'gem': GEM,
#     'cumulative': Cumulative,
#     'naive': Naive,
#     'joint': JointTraining,
#     'icarl': ICaRL
# }