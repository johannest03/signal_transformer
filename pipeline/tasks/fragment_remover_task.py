from ..components.task import Task
import pandas as pd
import ast
import numpy as np

class FragmentRemoverTask(Task):
    
    def process(self, idx, input, metadata):
        signal_fragments_df = pd.read_csv(metadata["signal_fragments"])
        fragment_intervals = signal_fragments_df[signal_fragments_df['PatientID'] == int(idx)].iloc[0]['Fragmentation_intervals']
        fragment_intervals = ast.literal_eval(fragment_intervals)
        
        fragment_parts = []
        
        for start, end in fragment_intervals:
            start_idx = start * metadata["frequency"]
            end_idx = len(input) if end == -1 else end * metadata["frequency"]
            fragment_parts.append((start_idx, end_idx))
            
        signal_segments = []
        last_end = 0
        for start, end in fragment_parts:
            signal_segments.append(input[last_end:start])
            last_end = end
        if last_end < len(input): # fragment ended before end of signal
            signal_segments.append(input[last_end:])
                                    
        return signal_segments