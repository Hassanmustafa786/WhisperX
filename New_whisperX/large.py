import requests
import json

# Load configuration from JSON file
with open('credentials.json', 'r') as config_file:
    config = json.load(config_file)

read_api_key = config['read_api']['api_token']
write_api_key =  config['write_api']['api_token']

import whisperx
import gc

device = "cpu"
audio_file = "audio.mp3"
batch_size = 8 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
# model_base = whisperx.load_model("large-v3", device, compute_type=compute_type)

# save model to local path (optional)
model_dir = "C:\\Users\\User\\.cache\whisper\\"
model_large = whisperx.load_model("large-v3", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model_large.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment
print("---------------------------------------------------------------------------------------------------")

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment
print("---------------------------------------------------------------------------------------------------")

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=write_api_key, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print("---------------------------------------------------------------------------------------------------")

print(result["segments"]) # segments are now assigned speaker IDs
print("---------------------------------------------------------------------------------------------------")
