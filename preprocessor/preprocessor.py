import os
import random
import json
import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import audio as Audio

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.stats_path = config["path"]["stats_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in ["phoneme_level", "frame_level"]
        assert config["preprocessing"]["energy"]["feature"] in ["phoneme_level", "frame_level"]
        self.pitch_phoneme_averaging = config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        self.energy_phoneme_averaging = config["preprocessing"]["energy"]["feature"] == "phoneme_level"

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    # New normalize method
    def normalize(self, directory, mean, std):
        min_val, max_val = float('inf'), float('-inf')
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            values = np.load(filepath)

            # Normalize values
            values = (values - mean) / std

            # Save normalized values
            np.save(filepath, values)

            # Update min and max
            min_val = min(min_val, np.min(values))
            max_val = max(max_val, np.max(values))

        return min_val, max_val

    def build_from_path(self):
        os.makedirs(os.path.join(self.out_dir, "mel"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pitch"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "energy"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "duration"), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        speakers = {}

        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(self.out_dir, "TextGrid", speaker, f"{basename}.TextGrid")
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                        out.append(info)

                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

                    n_frames += n

        print("Computing statistic quantities ...")
        pitch_mean, pitch_std = (pitch_scaler.mean_[0], pitch_scaler.scale_[0]) if self.pitch_normalization else (0, 1)
        energy_mean, energy_std = (energy_scaler.mean_[0], energy_scaler.scale_[0]) if self.energy_normalization else (0, 1)

        pitch_min, pitch_max = self.normalize(os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std)
        energy_min, energy_max = self.normalize(os.path.join(self.out_dir, "energy"), energy_mean, energy_std)

        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        stats = {
            "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
            "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
        }
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            f.write(json.dumps(stats))
        with open(os.path.join(self.stats_path, "stats.json"), "w") as f:
            f.write(json.dumps(stats))

        print("Total time: {} hours".format(n_frames * self.hop_length / self.sampling_rate / 3600))

        random.shuffle(out)
        out = [r for r in out if r is not None]

        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size:]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[:self.val_size]:
                f.write(m + "\n")

        return out


    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, f"{basename}.wav")
        text_path = os.path.join(self.in_dir, speaker, f"{basename}.lab")
        tg_path = os.path.join(self.out_dir, "TextGrid", speaker, f"{basename}.TextGrid")

        textgrid = tgt.io.read_textgrid(tg_path, encoding='utf-8-sig')
        phone_data = self.get_alignment(textgrid.get_tier_by_name("phones"))

        if phone_data is None:
            print(f"Skipping {basename} due to missing alignment data.")
            return None

        phone, duration, start, end = phone_data
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate * start):int(self.sampling_rate * end)].astype(np.float32)

        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        pitch, t = pw.dio(wav.astype(np.float64), self.sampling_rate, frame_period=self.hop_length / self.sampling_rate * 1000)
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[:sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, :sum(duration)]
        energy = energy[:sum(duration)]

        if self.pitch_phoneme_averaging:
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(nonzero_ids, pitch[nonzero_ids], fill_value="extrapolate", bounds_error=False)
            pitch = interp_fn(np.arange(0, len(pitch)))

            pos = 0
            for i, d in enumerate(duration):
                pitch[i] = np.mean(pitch[pos: pos + d]) if d > 0 else 0
                pos += d
            pitch = pitch[:len(duration)]

        if self.energy_phoneme_averaging:
            pos = 0
            for i, d in enumerate(duration):
                energy[i] = np.mean(energy[pos: pos + d]) if d > 0 else 0
                pos += d
            energy = energy[:len(duration)]

        np.save(os.path.join(self.out_dir, "duration", f"{speaker}-duration-{basename}.npy"), duration)
        np.save(os.path.join(self.out_dir, "pitch", f"{speaker}-pitch-{basename}.npy"), pitch)
        np.save(os.path.join(self.out_dir, "energy", f"{speaker}-energy-{basename}.npy"), energy)
        np.save(os.path.join(self.out_dir, "mel", f"{speaker}-mel-{basename}.npy"), mel_spectrogram.T)

        return (
            "|".join([basename, speaker, "{" + " ".join(phone) + "}", raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]
        phones, durations, start_time, end_time = [], [], 0, 0

        if not tier._objects:
            print("Warning: Empty tier data found.")
            return None

        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            if not phones and p in sil_phones:
                continue

            phones.append(p)
            duration = int(np.round((e - s) * self.sampling_rate / self.hop_length))
            durations.append(duration)
            end_time = e

        if len(phones) == 0:
            print("Warning: No phonemes found in TextGrid.")
            return None

        start_time = tier._objects[0].start_time if start_time == 0 else start_time
        end_time = tier._objects[-1].end_time

        return phones, durations, start_time, end_time

    def remove_outlier(self, values, threshold=1.5):
        values = np.array(values)
        if len(values) == 0:
            return values  # return empty if no values

        # Compute Q1 (25th percentile) and Q3 (75th percentile)
        Q1, Q3 = np.percentile(values[values > 0], [25, 75])  # Only consider non-zero values
        IQR = Q3 - Q1

        # Define the acceptable range and filter values
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_values = np.where((values > lower_bound) & (values < upper_bound), values, 0)

        return filtered_values
