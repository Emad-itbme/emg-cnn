import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


class EMGSingleFileFilter:
    """
    Filter a single merged EMG CSV file:
    - Band-pass (20–450 Hz)
    - Full-wave rectification
    - Low-pass filter (6 Hz) to get EMG envelope
    """

    def __init__(self, input_file, output_file, fs=1000):
        self.input_file = input_file
        self.output_file = output_file
        self.fs = fs

        # Filter settings
        self.bp_low = 20
        self.bp_high = 450
        self.lp_cutoff = 6

    def _butter_bandpass(self):
        low = self.bp_low / (self.fs / 2)
        high = self.bp_high / (self.fs / 2)
        return butter(4, [low, high], btype="band")

    def _butter_lowpass(self):
        cutoff = self.lp_cutoff / (self.fs / 2)
        return butter(4, cutoff, btype="low")

    def _filter_emg_signal(self, signal):
        """Apply band-pass → rectify → low-pass."""
        b, a = self._butter_bandpass()
        bandpassed = filtfilt(b, a, signal)

        rectified = np.abs(bandpassed)

        b2, a2 = self._butter_lowpass()
        envelope = filtfilt(b2, a2, rectified)

        return envelope

    def run(self):

        print(f"[+] Loading: {self.input_file}")
        df = pd.read_csv(self.input_file)

        filtered_df = pd.DataFrame()

        for col in df.columns:

            # Skip text columns
            if not np.issubdtype(df[col].dtype, np.number):
                print(f"[!] Skipping non-numeric column: {col}")
                filtered_df[col] = df[col]   # copy as is
                continue

            print(f"    Filtering: {col}")
            filtered_df[col] = self._filter_emg_signal(df[col].values)

        # Save output
        filtered_df.to_csv(self.output_file, index=False)
        print(f"[✓] Saved filtered EMG → {self.output_file}")


# ------------------------- RUN ---------------------------

if __name__ == "__main__":
    INPUT_FILE = r"C:\Users\Emadr\Desktop\dataset\H05\ALL_DELT_RAW.csv"
    OUTPUT_FILE = r"C:\Users\Emadr\Desktop\dataset\H05\ALL_DELT_FILTERED.csv"

    processor = EMGSingleFileFilter(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        fs=1000
    )

    processor.run()
