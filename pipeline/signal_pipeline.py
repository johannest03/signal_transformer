from .components.processing_stage import ProcessingStage
import pandas as pd
from tqdm import tqdm

class SignalPipeline():
    
    def __init__(self, metadata_path) -> None:
        self.processing_stages = []
        self.metadata = self._load_metadata(metadata_path)
        
    def _load_metadata(self, path):
        import pandas as pd
        return pd.read_csv(path)
        
    def add(self, processing_stage: ProcessingStage):
        self.processing_stages.append(processing_stage)
        
    def get_metadata(self):
        return self.metadata
        
    def run(self, idx, input=None):
        output = input
        if output is None:
            output = {}
        
        for processing_stage in self.processing_stages:
            for type in processing_stage.get_types():
                if type not in output:
                    output[type] = {}
                output[type] = processing_stage.process(idx, type, output[type], self.metadata)
        return output
    
    def run_all(self, classification=None):
        if classification is None:
            raise ValueError('classification is expected to be in a dataframe format')
        
        output = pd.DataFrame()
        for id in tqdm(classification['PatientID']):
            try:
                output[id] = [self.run(id)]
                output = output.copy()
            except Exception as e:
                print(f"Error for id: {id}", e)
        return output