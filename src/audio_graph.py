import random
from typing import List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException, UploadFile

# Initialise router
router = APIRouter()


def audio_to_df(
    audio_file: UploadFile,
    max_length: float = 1200,
    n_fft: int = 2048,
    n_mels: int = 256,
    hop_length: int = 512,
) -> pl.DataFrame:
    """Generate a df corresponding to a Mel spectrogram of an audio file.
    Columns are the Mel frequencies and rows are the timestamps."""

    # Load audio file
    waveform, sr = librosa.load(audio_file, sr=None, mono=True, duration=max_length)

    # Generate spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Compute the Mel frequencies
    frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr // 2).astype(
        int
    )

    # Create dataframe
    df = pl.from_numpy(spectrogram.transpose()).rename(
        {f"column_{i}": f"{f}" for i, f in enumerate(frequencies.astype(str))}
    )

    return df


def filter_silence(df: pl.DataFrame, threshold_db: float = -50.0):
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


def compute_spectrum(df: pl.DataFrame) -> pl.DataFrame:
    """Compute the mean of the Mel spectrogram. In particular,
    it returns the vector of the mean of each Mel frequency as numpy
    arrays."""

    # Compute the mean
    df = df.mean()

    # Get the frequencies
    frequencies = np.asarray(df.columns).astype(int)

    # Get the actual means
    means = df.to_numpy().reshape(-1)

    return means, frequencies


def generate_random_color():
    """Generate a random HEX color to be used in the graph"""
    return "#" + "".join([random.choice("0123456789ABCDEF") for i in range(6)])


def plot_eq(
    frequencies: list, spectra: list, tracks: list, threshold_db: float = -50.0
):
    # Init the figure
    fig = plt.figure(figsize=(20, 10))

    # Create a graph for every instrument
    for instrument, frequency, spectrum in zip(tracks, frequencies, spectra):
        # Pick a random color
        color = generate_random_color()

        # Plot the frequency response
        plt.plot(frequency, spectrum, label=instrument, color=color)

        # Fill the area below the threshold
        plt.fill_between(frequency, threshold_db, spectrum, alpha=0.2, color=color)

        # Use a log scale on the x axis
        plt.xscale("log")

        # Define custom ticks to mimic e.g. FabFilter Pro-Q 3
        ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        plt.xticks(ticks, [str(tick) for tick in ticks])

        # Add a legend
        plt.legend(fontsize="large")

        # Name the axis
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain (dB)")

    return fig

    
from fastapi import  Form


@router.post("/eq")
def generate_eq(
    audio_files: List[UploadFile],
    tracks: List[str] = Form(...),
    max_length: float = 1200,
    max_files: int = 5,
    threshold_db: float = -50.0,
    normalize=False,
):
    """Create an EQ graph for a list of audio files and a list of instruments.
    The audio files must be in the same order as the instruments."""

    # Exceed number of admissible audio files to be processed
    # in batch
    if len(audio_files) > max_files:
        # Create error message
        ERROR = {
            "code": "413",
            "type": "payload too large",
            "error": "too many files uploaded at once",
            "message": f"resubmit request with no more than {max_files} files",
        }

        raise HTTPException(status_code=413, detail=ERROR)

    # Compute the mean of the spectrogram for each audio file
    spectra_and_frequencies = [
        compute_spectrum(
            filter_silence(
                audio_to_df(audio_file.file, max_length=max_length),
                threshold_db=threshold_db,
            )
        )
        for audio_file in audio_files
    ]

    # Disentangle
    spectra, frequencies = zip(*spectra_and_frequencies)

    # If normalize, normalize the spectra
    if normalize:
        spectra = (
            spectra - np.nanmax(spectra, axis=1, keepdims=True) + np.nanmax(spectra)
        )

    # Plot the EQ
    fig = plot_eq(frequencies, spectra, tracks, threshold_db=threshold_db)

    return {"eq": "ok"}
