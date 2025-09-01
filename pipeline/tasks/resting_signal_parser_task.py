from ..components.task import Task
from ..parser.resting_signal_parser import RestingSignalParser
import numpy as np

class RestingSignalParserTask(Task, RestingSignalParser):
    def process(self, idx, input, metadata):
        return np.array(self.get_resting_signal(idx, metadata))