from .signal_pipeline import SignalPipeline
import pandas as pd

class FeaturePipelineRunner():
    
    def __init__(self, pipeline: SignalPipeline, save_path, CHECK_IF_CSV_EXISTS=True) -> None:
        self.pipeline = pipeline
        self.save_path = save_path
        self.output = None
        self.CHECK_IF_CSV_EXISTS = CHECK_IF_CSV_EXISTS
    
    def save(self):
        if self.output is None:
            return
        self.output.to_csv(self.save_path, index=False)
    
    def _bundle_result(self, idx, input):
        records = []
        for _, row in input.iterrows():
            signals = row[idx]
            n_segments = max(len(v) for v in signals.values())
            
            for i in range(n_segments):
                rec = {"PatientID": idx, "segment":i}
                for sig, seg_list in signals.items():
                    if i < len(seg_list):
                        for k, v in seg_list[i].items():
                            rec[k] = v
                records.append(rec)
                
        flat_df = pd.DataFrame(records)
        return flat_df
        
    def get(self, idx):
        output = self.pipeline.run(idx=idx)
        self.output = self._bundle_result(idx, output)
        return self.output
        
    def get_all(self, classification):
        if self.CHECK_IF_CSV_EXISTS:
            try:
                return pd.read_csv(self.save_path)
            except:
                pass
        
        output = self.pipeline.run_all(classification)
        final_output = pd.DataFrame()
        for idx in output.columns:
            final_output = pd.concat([final_output, self._bundle_result(idx, output)], ignore_index=True)
        self.output = final_output
        self.save()
        return self.output
        
    def get_metadata(self):
        return self.pipeline.get_metadata()
            