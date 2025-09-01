from ..enum_signal_type import SignalType
from .task import Task
class ProcessingStage():
    
    def __init__(self):
        self.tasks = {}
        
    def add(self, task:Task, signal_types = [SignalType.GLOBAL]): 
        for signal_type in signal_types:
            if signal_type not in self.tasks:
                self.tasks[signal_type] = [task]
            else:
                self.tasks[signal_type].append(task)
            
    def process(self, idx, signal_type:SignalType, input, meta):
        tasks = self._get_tasks(signal_type)
        for task in tasks:
            input = task.process(idx, input, meta[meta['type'] == signal_type.name].iloc[0])
        return input
            
    def _get_tasks(self, signal_type:SignalType):
        tasks = []
        if SignalType.GLOBAL in self.tasks:
            tasks = self.tasks[SignalType.GLOBAL]
        elif signal_type in self.tasks:
            tasks.extend(self.tasks[signal_type])
        return tasks
    
    def get_types(self):
        signal_types = set()
        for signal_type in self.tasks:
            if signal_type is SignalType.GLOBAL:
                signal_types.update(list(SignalType))
            signal_types.add(signal_type)
        if SignalType.GLOBAL in signal_types:
            signal_types.remove(SignalType.GLOBAL)
        return signal_types
    