from .feature_extractor import FeatureExtractor
import hrvanalysis  as hrv
import neurokit2 as nk
import pandas as pd


class ECGFeatureExtractor(FeatureExtractor):
    def extract_features(self,idx, signal, frequency):
        nn_intervals_list, r_peaks = self.get_nn_intervall_list(signal=signal, frequency=frequency)
        features = dict()
        for key, value in hrv.get_time_domain_features(nn_intervals_list).items():
            features[key] = value
        for key, value in hrv.get_frequency_domain_features(nn_intervals_list).items():
            features[key] = value
        for key, value in hrv.get_csi_cvi_features(nn_intervals_list).items():
            features[key] = value
        for key, value in hrv.get_poincare_plot_features(nn_intervals_list).items():
            features[key] = value
        return features
    
    def get_nn_intervall_list(self, signal, frequency):
        _, info = nk.ecg_process(signal, sampling_rate=frequency)
        r_peaks = info['ECG_R_Peaks']
        rr_peaks_intervall = []
        for i in range(0, len(r_peaks)-1):
            rr_peaks_intervall.append(r_peaks[i+1]-r_peaks[i])

        rr_peaks_intervall = hrv.remove_outliers(rr_intervals=rr_peaks_intervall,  low_rri=500, high_rri=1500, verbose=False)

        rr_peaks_intervall = hrv.interpolate_nan_values(rr_intervals=rr_peaks_intervall, interpolation_method="linear")

        nn_intervals_list = hrv.remove_ectopic_beats(rr_intervals=rr_peaks_intervall, method="malik", verbose=False)
        nn_intervals_list = hrv.interpolate_nan_values(rr_intervals=nn_intervals_list)
            
        return nn_intervals_list, r_peaks