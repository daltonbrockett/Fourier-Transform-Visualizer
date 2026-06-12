# Fourier Transform Visualizer

This is a Python program that takes an audio or video file as input, extracts the audio, computes its Fourier Transform, and outputs either a 2D `.png` photo or a 3D printable `.stl` file!

## Setup

1. **Install Python 3** (if you haven't already).
2. **Install the dependencies**. In your terminal, run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script from your terminal and provide the path to your audio or video file:

```bash
python visualizer.py path/to/your/video.mp4
```

To specify a custom output filename (the script will automatically use `.png` or `.stl` depending on your mode):
```bash
python visualizer.py path/to/your/audio.wav -o awesome_model
```

## Visualization Modes

By default, the script generates a 2D plot. To generate a 3D model instead, use the `--3d` flag:

- **2D Mode**: `python visualizer.py path/to/your/audio.wav` generates the standard 2D frequency spectrum. Saves as a `.png` photo.
- **3D Mode**: `python visualizer.py path/to/your/audio.wav --3d` generates a continuous 3D-printable solid block where the front profile perfectly matches the 2D spectrum, and the frequency peaks ripple backwards into the Time dimension as continuous sine waves. Saves as a `.stl` file!

## Customization

Open `visualizer.py` in a text editor to easily tweak the configuration variables at the top of the file:

- **`MIN_FREQ` and `MAX_FREQ`**: Adjust the "window" of frequencies processed here.
- **`LOG_SCALE`**: Set to `True` to use a logarithmic scale, which makes smaller, quieter harmonic frequencies much more visible.
- **`MAGNITUDE_CUTOFF`**: A value from 0.0 to 1.0. Chops off the bottom X% of the signal's background noise so the peaks stick out more prominently from the base. Try 0.2 to 0.4 if `LOG_SCALE` is too noisy!
- **2D Colors**: Change `BACKGROUND_COLOR`, `LINE_COLOR`, and `FILL_COLOR` (you can use hex codes like `#ff0055`) for the 2D photo mode.
- **3D Dimensions**: You can modify `WIDTH_MM`, `DEPTH_MM`, and `BASE_THICKNESS_MM` to ensure the final `.stl` file perfectly matches your 3D printer's build volume. You can also change `MAX_HEIGHT_MM` to control how tall the mountains/waves are, and tweak `RESOLUTION_X` and `RESOLUTION_Y` for print quality!