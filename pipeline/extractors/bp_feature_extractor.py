from .feature_extractor import FeatureExtractor
from pyPPG.datahandling import load_data
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
from pyPPG import PPG, Fiducials
from ...parser.signal_parser import SignalParser 
import os
import pandas as pd
from ..feature_calculators.bp_area_features import BPAreaFeatures
from utils.hidden_prints import HiddenPrints

import warnings # igonore all pandas warning that FP throws as its outa
warnings.filterwarnings('ignore')

class BPFeatureExtractor(FeatureExtractor):
    
    def __init__(self) -> None:
        super().__init__()
        self.area_feature_calculator = BPAreaFeatures()
    
    def extract_features(self, idx, signal, frequency):
        """Extracts the mean values of the bp"""
        with HiddenPrints():
            # write signal to txt file and then load it with pyPPG method :D
            parser = SignalParser()
            tmp_file = './bp_tmp.txt'
            parser.write_to_txt_file(path=tmp_file, signal=signal)
            pyppg_signal = load_data(data_path=tmp_file, fs=frequency)
            os.remove(tmp_file)

            pyppg_signal.filtering = True
            pyppg_signal.fL=0.5000001 # Lower cutoff frequency (Hz)
            pyppg_signal.fH=12 # Upper cutoff frequency (Hz)
            pyppg_signal.order=4 # Filter order
            pyppg_signal.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10} # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"

            prep = PP.Preprocess(fL=pyppg_signal.fL, fH=pyppg_signal.fH, order=pyppg_signal.order, sm_wins=pyppg_signal.sm_wins)
            pyppg_signal.ppg, pyppg_signal.vpg, pyppg_signal.apg, pyppg_signal.jpg = prep.get_signals(s=pyppg_signal)

            corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
            correction=pd.DataFrame()
            correction.loc[0, corr_on] = True
            pyppg_signal.correction=correction

            # Create a PPG class
            s = PPG(pyppg_signal)

            # Initialise the fiducials package
            fpex = FP.FpCollection(s=s)
            fiducials = fpex.get_fiducials(s=s)
            
            # Create a fiducials class
            fp = Fiducials(fp=fiducials)

            # Init the biomarkers package
            bmex = BM.BmCollection(s=s, fp=fp)

            # Extract biomarkers
            _, _, bm_stats = bmex.get_biomarkers()

            bm_stats = pd.DataFrame.to_dict(bm_stats['ppg_sig'])

            features = {}

            for key,value in bm_stats.items():
                features[key] = value['mean']

            # overrite the area feature as they are wrong in the lib        
            onset_indexes = fp.get_fp()['on']
            discrotic_notch_indexes = fp.get_fp()['dn']
            features['AUCpi'] = self.area_feature_calculator.get_avg_AUCpi(signal=signal, onset_indexes=onset_indexes, frequency = frequency)
            features['AUCsys'] = self.area_feature_calculator.get_avg_AUCsys(signal=signal, onset_indexes=onset_indexes, discrotic_notch_indexes=discrotic_notch_indexes, frequency = frequency)
            features['AUCdia'] = self.area_feature_calculator.get_avg_AUCdia(signal=signal, onset_indexes=onset_indexes, discrotic_notch_indexes=discrotic_notch_indexes, frequency = frequency)
        
        return features
    
   