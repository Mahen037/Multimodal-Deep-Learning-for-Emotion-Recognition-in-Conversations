## Generate wav files from mp4 files
  # The wav files are saved in the same directory as the mp4 files
  # The wav files are named the same as the mp4 files, but with the .wav extension
  # The wav files are 16000 Hz, 16 bit, mono
  # Install ffmpeg if not already installed - brew install ffmpeg

## To be run in the terminal
for f in mini-test/MELD-RAW/MELD.Raw/train/dia*_utt*.mp4; do
    b=$(basename "$f" .mp4)
    ffmpeg -y -i "$f" -ac 1 -ar 16000 "mini-test/MELD-RAW/MELD.Raw/train/$b.wav"
  done