from .feature_extractor import FeatureExtractor
import pandas as pd

class CSVFeatureExtractor(FeatureExtractor):

    def __init__(self, metadata) -> None:
        super().__init__()
        self.csv_file = metadata[metadata['type'] == 'CSV'].iloc[0]['file_name']


    def extract_features(self, idx, signal, frequency):
        print("HELLO????")
        data = pd.read_csv(self.csv_file)
        print("CSV DATA: ", data)
        data = data[data['PatientID'] == int(idx)]
        data = data.drop(['PatientID'], axis=1)
        print("CSV DATA: ", data)
        return data.squeeze().to_dict()