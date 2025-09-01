from ..components.task import Task

class SignalSegmentationTask(Task):
    def process(self, idx, input, metadata):
        signals = []
        
        window_size=metadata['window_size'] * metadata['frequency']
        window_stride = metadata['window_stride'] * metadata['frequency']
        for signal in input:
            for start in range(0, len(signal), window_stride):
                if start + window_size >= len(signal):
                    break
                signals.append(signal[start:(start+window_size)])
        
        return signals