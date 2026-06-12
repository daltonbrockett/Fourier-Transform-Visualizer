import argparse
import numpy as np
import matplotlib.pyplot as plt
from moviepy import AudioFileClip
import scipy.signal
from stl import mesh

# --- Configuration Variables ---
# --- 3D Printing Settings ---
# Physical dimensions of the final STL file (in mm)
WIDTH_MM = 100.0          # X-axis: Frequency spread
DEPTH_MM = 100.0          # Y-axis: Time spread
MAX_HEIGHT_MM = 20.0      # Z-axis: Max amplitude
BASE_THICKNESS_MM = 2.0   # Thickness of the solid base layer
TIME_LENGTH = 0.01        # How much real time the depth axis represents (seconds)
RESOLUTION_X = 300       # Number of frequency bins to sample
RESOLUTION_Y = 300        # Number of time steps to sample
SMOOTHING_SIGMA = 2.0     # Smooths out sharp peaks (0.0 to disable)

# Adjust these variables to change the output!

# --- Audio Processing Settings ---
# If True, applies a logarithmic scale to the magnitudes.
# This makes smaller, quieter harmonic frequencies much more visible.
LOG_SCALE = False

# Chops off the bottom X% of the signal to remove background noise/low-level harmonics.
# This brings the peaks back down to the base surface when LOG_SCALE is True.
# A value between 0.0 (no cutoff) and 1.0 (get rid of everything).
MAGNITUDE_CUTOFF = 0.0

# Frequency range to process in Hz
MIN_FREQ = 0
MAX_FREQ = 4000

# --- Settings for 2D matplotlib ---
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
        clip.close()
    
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
    fft_result = np.fft.rfft(audio_data)
    magnitudes = np.abs(fft_result) / n_samples
    frequencies = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    return frequencies, magnitudes

def apply_log_and_normalize(magnitudes):
    """
    Applies logarithmic scaling (if enabled) and normalizes the magnitudes 
    so the tallest peak in the window is exactly 1.0.
    """
    if LOG_SCALE:
        magnitudes = 20 * np.log10(magnitudes + 1e-9)
    
    # Normalize between 0.0 and 1.0
    magnitudes = (magnitudes - np.min(magnitudes)) / (np.max(magnitudes) - np.min(magnitudes) + 1e-9)
    
    # Chop off the bottom noise floor so peaks protrude from the base surface
    if MAGNITUDE_CUTOFF > 0.0:
        magnitudes = np.clip(magnitudes - MAGNITUDE_CUTOFF, 0, None)
        # Re-normalize so the tallest peak is back to 1.0
        if np.max(magnitudes) > 0:
            magnitudes = magnitudes / np.max(magnitudes)
            
    return magnitudes

def create_visualization(audio_data, sample_rate, output_path):
    print(f"Generating 2D output...")
    
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    frequencies, magnitudes = perform_fft(audio_data, sample_rate)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BACKGROUND_COLOR)
    mask = (frequencies >= MIN_FREQ) & (frequencies <= MAX_FREQ)
    
    freqs_to_plot = frequencies[mask]
    mags_to_plot = apply_log_and_normalize(magnitudes[mask])
    
    ax.plot(freqs_to_plot, mags_to_plot, color=LINE_COLOR, linewidth=LINE_WIDTH)
    ax.fill_between(freqs_to_plot, mags_to_plot, color=FILL_COLOR, alpha=FILL_ALPHA)
    ax.set_xlim(MIN_FREQ, MAX_FREQ)
    ax.set_xlabel('Frequency (Hz)', color='white')
    
    y_label = 'Magnitude (Log Scale)' if LOG_SCALE else 'Magnitude'
    ax.set_ylabel(y_label, color='white')
    
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_alpha(0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Done! Saved photo to {output_path}")

def create_3d_model(audio_data, sample_rate, output_path):
    print(f"Generating 3D output...")
    frequencies, magnitudes = perform_fft(audio_data, sample_rate)
    
    mask = (frequencies >= MIN_FREQ) & (frequencies <= MAX_FREQ)
    freqs_to_plot = frequencies[mask]
    mags_to_plot = apply_log_and_normalize(magnitudes[mask])
    
    if len(freqs_to_plot) == 0:
        print("No frequencies in the specified range.")
        return

    # Subsample or interpolate frequencies to match RESOLUTION_X
    x_new = np.linspace(MIN_FREQ, MAX_FREQ, RESOLUTION_X)
    z_mag = np.interp(x_new, freqs_to_plot, mags_to_plot)
    
    if SMOOTHING_SIGMA > 0:
        from scipy.ndimage import gaussian_filter1d
        z_mag = gaussian_filter1d(z_mag, sigma=SMOOTHING_SIGMA)
    
    # Create time steps
    t_steps = np.linspace(0, TIME_LENGTH, RESOLUTION_Y)
    
    # Build Vertices
    vertices = []
    
    for y_idx in range(RESOLUTION_Y):
        t = t_steps[y_idx]
        y_pos = (y_idx / (RESOLUTION_Y - 1)) * DEPTH_MM
        for x_idx in range(RESOLUTION_X):
            f = x_new[x_idx]
            x_pos = (x_idx / (RESOLUTION_X - 1)) * WIDTH_MM
            z_pos = max(0, z_mag[x_idx] * np.cos(2 * np.pi * f * t) * MAX_HEIGHT_MM)
            vertices.append([x_pos, y_pos, z_pos])
            
    offset = RESOLUTION_X * RESOLUTION_Y
    for y_idx in range(RESOLUTION_Y):
        y_pos = (y_idx / (RESOLUTION_Y - 1)) * DEPTH_MM
        for x_idx in range(RESOLUTION_X):
            x_pos = (x_idx / (RESOLUTION_X - 1)) * WIDTH_MM
            z_pos = -BASE_THICKNESS_MM
            vertices.append([x_pos, y_pos, z_pos])
            
    vertices = np.array(vertices)
    faces = []
    
    def top_idx(x, y): return y * RESOLUTION_X + x
    def bot_idx(x, y): return offset + y * RESOLUTION_X + x

    # Top surface
    for y in range(RESOLUTION_Y - 1):
        for x in range(RESOLUTION_X - 1):
            v0 = top_idx(x, y)
            v1 = top_idx(x+1, y)
            v2 = top_idx(x, y+1)
            v3 = top_idx(x+1, y+1)
            faces.extend([[v0, v1, v2], [v1, v3, v2]])
            
    # Bottom surface
    for y in range(RESOLUTION_Y - 1):
        for x in range(RESOLUTION_X - 1):
            v0 = bot_idx(x, y)
            v1 = bot_idx(x+1, y)
            v2 = bot_idx(x, y+1)
            v3 = bot_idx(x+1, y+1)
            faces.extend([[v0, v2, v1], [v1, v2, v3]])

    # Front wall (y = 0)
    for x in range(RESOLUTION_X - 1):
        v0_t = top_idx(x, 0)
        v1_t = top_idx(x+1, 0)
        v0_b = bot_idx(x, 0)
        v1_b = bot_idx(x+1, 0)
        faces.extend([[v0_t, v0_b, v1_t], [v1_t, v0_b, v1_b]])
        
    # Back wall (y = RESOLUTION_Y - 1)
    for x in range(RESOLUTION_X - 1):
        v0_t = top_idx(x, RESOLUTION_Y - 1)
        v1_t = top_idx(x+1, RESOLUTION_Y - 1)
        v0_b = bot_idx(x, RESOLUTION_Y - 1)
        v1_b = bot_idx(x+1, RESOLUTION_Y - 1)
        faces.extend([[v0_t, v1_t, v0_b], [v1_t, v1_b, v0_b]])

    # Left wall (x = 0)
    for y in range(RESOLUTION_Y - 1):
        v0_t = top_idx(0, y)
        v1_t = top_idx(0, y+1)
        v0_b = bot_idx(0, y)
        v1_b = bot_idx(0, y+1)
        faces.extend([[v0_t, v1_t, v0_b], [v1_t, v1_b, v0_b]])
        
    # Right wall (x = RESOLUTION_X - 1)
    for y in range(RESOLUTION_Y - 1):
        v0_t = top_idx(RESOLUTION_X - 1, y)
        v1_t = top_idx(RESOLUTION_X - 1, y+1)
        v0_b = bot_idx(RESOLUTION_X - 1, y)
        v1_b = bot_idx(RESOLUTION_X - 1, y+1)
        faces.extend([[v0_t, v0_b, v1_t], [v1_t, v0_b, v1_b]])

    faces = np.array(faces)
    
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :]
            
    cube.save(output_path)
    print(f"Done! Saved 3D model to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a frequency domain visualization from an audio/video file.")
    parser.add_argument("input_file", help="Path to the input file.")
    parser.add_argument("-o", "--output", default="output", help="Path to save the output file.")
    parser.add_argument("--3d", action="store_true", dest="is_3d", help="Generate a 3D STL file instead of a 2D plot.")
    args = parser.parse_args()
    
    try:
        audio, sr = extract_audio(args.input_file)
        if args.is_3d:
            if not args.output.lower().endswith('.stl'):
                args.output += '.stl'
            create_3d_model(audio, sr, args.output)
        else:
            if not args.output.lower().endswith('.png'):
                args.output += '.png'
            create_visualization(audio, sr, args.output)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
