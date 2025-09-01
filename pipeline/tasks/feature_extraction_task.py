from ..components.task import Task
from ..extraction.extractors.feature_extractor import FeatureExtractor
import numpy as np

class FeatureExtractionTask(Task):
    
    def __init__(self, extractor: FeatureExtractor) -> None:
        super().__init__()
        self.extractor = extractor
        
    
    def process(self, idx, input, metadata):
        features = []
        for signal in input:
            # TODO fix this issue with csv, input is none but the csv has to return as many arrays as the bp signal
            features.append(self.extractor.extract_features(idx=idx, signal=signal, frequency=metadata['frequency'])) 
        return features