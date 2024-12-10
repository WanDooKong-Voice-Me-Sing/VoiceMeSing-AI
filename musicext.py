from pytubefix import YouTube
from pytubefix.cli import on_progress
import subprocess
import os

# Get YouTube URL from the user
url = input("Enter YouTube URL: ")
yt = YouTube(url, on_progress_callback=on_progress)

print(f"Downloading: {yt.title}")

# Define output directory
output_dir = "source/data"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Download the audio file
audio_stream = yt.streams.get_audio_only()
downloaded_file = os.path.join(output_dir, audio_stream.default_filename)

audio_stream.download(output_path=output_dir)
print(f"Downloaded file: {downloaded_file}")

# Determine FLAC file path
flac_file = os.path.splitext(downloaded_file)[0] + ".flac"

# Convert the downloaded file to FLAC format using ffmpeg
try:
    subprocess.run(
        ["ffmpeg", "-y", "-i", downloaded_file, flac_file],
        check=True, capture_output=True, text=True
    )
    print(f"FLAC file created successfully: {flac_file}")
except subprocess.CalledProcessError as e:
    print(f"Error during conversion: {e.stderr}")
finally:
    # Optionally remove the original file
    if os.path.exists(downloaded_file):
        os.remove(downloaded_file)
        print(f"Temporary file removed: {downloaded_file}")
