# import requests
# import json

# # Load configuration from JSON file
# with open('credentials.json', 'r') as config_file:
#     config = json.load(config_file)

# read_api_key = config['read_api']['api_token']
# write_api_key =  config['write_api']['api_token']

# import whisperx
# import gc
# import streamlit as st

# device = "cpu"
# audio_file = "audio.mp3"
# batch_size = 8 
# compute_type = "int8"

# model_dir = "C:\\Users\\User\\.cache\whisper\\"
# model_tiny = whisperx.load_model("tiny", device, compute_type=compute_type, download_root=model_dir)

# audio = whisperx.load_audio(audio_file)
# result = model_tiny.transcribe(audio, batch_size=batch_size)

# model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
# result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# diarize_model = whisperx.DiarizationPipeline(use_auth_token=write_api_key, device=device)

# diarize_segments = diarize_model(audio)

# result = whisperx.assign_word_speakers(diarize_segments, result)

# for segment in result['segments']:
#     print(segment['speaker'] + ": " + segment['text'])




import json
import whisperx
import streamlit as st

# Load configuration from JSON file
with open('credentials.json', 'r') as config_file:
    config = json.load(config_file)

read_api_key = config['read_api']['api_token']
write_api_key = config['write_api']['api_token']

st.set_page_config(layout='wide')

# Streamlit UI
st.title("Speech Transcription and Diarization")

# Upload audio file
audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "flac"])

if audio_file:
    st.audio(audio_file)

    # Load models
    device = "cpu"
    batch_size = 4
    compute_type = "int8"
    # model_dir = "C:\\Users\\User\\.cache\whisper\\"  # Modify this as per your system

    # Load large model
    model_tiny = whisperx.load_model("tiny", device, compute_type=compute_type)

    # Load audio
    audio = whisperx.load_audio(audio_file.name)

    # Transcribe audio
    st.write("Transcribing audio...")
    result = model_tiny.transcribe(audio, batch_size=batch_size)

    # Align segments
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # Diarize segments
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=write_api_key, device=device)
    st.write("Performing diarization...")
    diarize_segments = diarize_model(audio)

    # Assign word speakers
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Display transcribed text with speakers
    st.subheader("Transcribed Text with Speakers:")
    for segment in result['segments']:
        st.success(f"{segment['speaker']}: {segment['text']}")
