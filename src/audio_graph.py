import librosa
import numpy as np
import polars as pl


def audio_to_df(
    audio_file: str, n_fft: int = 2048, n_mels: int = 256, hop_length: int = 512
) -> pl.DataFrame:
    """Generate a df corresponding to a Mel spectrogram of an audio file.
    Columns are the Mel frequencies and rows are the timestamps."""

    # Load audio file
    waveform, sr = librosa.load(audio_file, sr=None, mono=True)

    # Generate spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Compute the Mel frequencies
    frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr // 2).astype(
        int
    )

    # # Compute timestamps
    # timestamps = np.arange(0, len(waveform), hop_length)

    # # Fix the last entry of the timestamp if necessary
    # if len(timestamps) == len(waveform) - 1:
    #     timestamps = np.append(timestamps, len(waveform))

    # # Convert to seconds
    # timestamps = timestamps * 1 / sr

    # Create dataframe
    df = pl.from_numpy(spectrogram.transpose()).rename(
        {f"column_{i}": f"{f}" for i, f in enumerate(frequencies.astype(str))}
    )

    # # Add timestamps and reorder columns
    # df = df.with_columns(pl.lit(timestamps).alias("timestamp")).select(
    #     pl.col("timestamp"), pl.all().exclude("timestamp")
    # )

    return df


def filter_silence(df: pl.DataFrame, threshold_db: float = -50):
    """Filter out silent frames from a spectrogram dataframe.
    The threshold is in decibels."""

    return df.with_columns(
        [
            pl.when(pl.col(col) < threshold_db)
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in df.columns
        ]
    )

def compute_metric(df: pl.DataFrame) -> pl.DataFrame:
    
    """ Compute the mean of the Mel spectrogram. In particular,
    it returns the vector of the mean of each Mel frequency as numpy 
    arrays."""

    # Compute the mean
    df = df.mean()
    
    # Get the frequencies
    frequencies = np.asarray(df.columns).astype(int)
    
    # Get the actual means
    means = df.to_numpy().reshape(-1)
    
    return means, frequencies
    