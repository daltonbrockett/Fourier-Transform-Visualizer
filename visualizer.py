import argparse
import numpy as np
import matplotlib.pyplot as plt
from moviepy import AudioFileClip

# --- Configuration Variables ---
# Adjust these variables to change the output!

# Frequency range to display on the graph (in Hertz).
# Human hearing is roughly 20 Hz to 20,000 Hz. 
# Viola ranges typically sit between 130 Hz (C3) and a few thousand Hz for overtones.
MIN_FREQ = 0
MAX_FREQ = 3000

# Settings for matplotlib
BACKGROUND_COLOR = '#0f0f0f'
LINE_COLOR = '#00ffcc'
FILL_COLOR = '#00ffcc'
FILL_ALPHA = 0.3
LINE_WIDTH = 1.0

def extract_audio(file_path):
    """
    Extracts mono audio from a given video or audio file.
    Supports .wav, .mp3, .mp4, .mov, etc. via MoviePy.
    """
    print(f"Loading {file_path}...")
    # to_soundarray returns a numpy array. 
    # If it's stereo, it has shape (N, 2). If mono, (N,) or (N, 1).
    if file_path.lower().endswith('.wav'):
        import scipy.io.wavfile as wavfile
        sample_rate, audio_array = wavfile.read(file_path)
        # Normalize audio to [-1.0, 1.0] to match moviepy output
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_array.dtype == np.int32:
            audio_array = audio_array.astype(np.float32) / 2147483648.0
        elif audio_array.dtype == np.uint8:
            audio_array = (audio_array.astype(np.float32) - 128) / 128.0
    else:
        clip = AudioFileClip(file_path)
        sample_rate = clip.fps
        audio_array = clip.to_soundarray(fps=sample_rate)
        # Free up resources
        clip.close()
    
    
    # Convert to mono by averaging the two channels if it's stereo
    if audio_array.ndim > 1 and audio_array.shape[1] > 1:
        print("Converting stereo to mono...")
        audio_array = np.mean(audio_array, axis=1)
    elif audio_array.ndim > 1:
        audio_array = audio_array.squeeze()
        
    return audio_array, sample_rate

def perform_fft(audio_data, sample_rate):
    """
    Performs a Fast Fourier Transform on the full audio data.
    """
    print("Performing Fourier Transform...")
    n_samples = len(audio_data)
    
    # We use rfft (Real FFT) because our input is purely real audio data,
    # and rfft is faster and only returns the positive frequencies.
    fft_result = np.fft.rfft(audio_data)
    
    # Compute the magnitude (amplitude) of each frequency.
    # We divide by n_samples to normalize the amplitudes.
    magnitudes = np.abs(fft_result) / n_samples
    
    # Get the corresponding frequencies (x-axis values)
    frequencies = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    
    return frequencies, magnitudes

def create_visualization(frequencies, magnitudes, output_path):
    """
    Plots the frequency domain and saves it as a PNG.
    """
    print(f"Generating visualization: {output_path}")
    
    # Setup matplotlib figure
    # You can change the figsize (width, height) in inches
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Filter the data to only include the frequencies within our MIN_FREQ and MAX_FREQ bounds
    # This makes plotting faster and focuses the graph on what you care about.
    mask = (frequencies >= MIN_FREQ) & (frequencies <= MAX_FREQ)
    freqs_to_plot = frequencies[mask]
    mags_to_plot = magnitudes[mask]
    
    # Plot the line
    ax.plot(freqs_to_plot, mags_to_plot, color=LINE_COLOR, linewidth=LINE_WIDTH)
    
    # Fill the area under the curve for an artistic effect
    ax.fill_between(freqs_to_plot, mags_to_plot, color=FILL_COLOR, alpha=FILL_ALPHA)
    
    # Customize the axes
    ax.set_xlim(MIN_FREQ, MAX_FREQ)
    ax.set_xlabel('Frequency (Hz)', color='white', fontsize=12)
    ax.set_ylabel('Magnitude', color='white', fontsize=12)
    
    # Color the ticks and spines (borders) white
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_alpha(0.3)
        
    # Optional: You can hide the axes:
    # To hide everything (both axes, ticks, and borders):
    # ax.axis('off')
    
    # just the x-axis:
    # ax.xaxis.set_visible(False)
    
    # just y-axis:
    # ax.yaxis.set_visible(False)
    
    ax.axis('off') # Currently hiding everything, you can change this!
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Generate a frequency domain visualization from an audio/video file.")
    parser.add_argument("input_file", help="Path to the input .wav, .mp3, .mp4, or .mov file.")
    parser.add_argument("-o", "--output", default="output.png", help="Path to save the output .png image.")
    args = parser.parse_args()
    
    try:
        audio, sr = extract_audio(args.input_file)
        freqs, mags = perform_fft(audio, sr)
        create_visualization(freqs, mags, args.output)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
