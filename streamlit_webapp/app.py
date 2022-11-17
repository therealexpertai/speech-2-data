import librosa
import torch
import time
import datetime
from pathlib import Path
import subprocess
import os
import shutil
import soundfile as sf
import streamlit as st
from PIL import Image
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datetime import date
from expertai.nlapi.cloud.client import ExpertAiClient
from creds import *

st.set_page_config(
    page_title="Speech <2> Data",
    page_icon="🗣",
    layout="centered",
    initial_sidebar_state="auto",
)

main_image = Image.open('static/main_banner.png')

sr = 16000
block_length = 30
language = "en"

path_base = "Audio files/"
audio_report = "Reports"
transcripts = "transcripts"
path_converted_audio = "converted_files/"
resampled_folder = "resampled_files/"

Path(path_base).mkdir(parents = True, exist_ok = True)
Path(audio_report).mkdir(parents = True, exist_ok = True)
Path(transcripts).mkdir(parents = True, exist_ok = True)
Path(path_converted_audio).mkdir(parents = True, exist_ok = True)
Path(resampled_folder).mkdir(parents = True, exist_ok = True)

extension_to_convert = ['.mp3','.mp4','.m4a','.flac','.opus']

@st.cache(show_spinner=False,suppress_st_warning=True)
def clean_directory(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.abspath(os.path.join(dirpath, filename))
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

@st.cache(show_spinner=False,suppress_st_warning=True)
def instantiate_model():
    model = "facebook/wav2vec2-base-960h"
    print("Loading model: ", model)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return model, processor


def preprocessing(path_base, path_converted_audio):
    for file in os.listdir(path_base):
        filename, file_extension = os.path.splitext(file)
        print("\nFile name: " + file)

        if file_extension == ".wav":
            file_to_process = file
            shutil.copy(path_base + file, path_converted_audio + file)
        elif file_extension in extension_to_convert:
            subprocess.call(['ffmpeg', '-i', path_base + file,
            path_base + filename + ".wav"])
            shutil.move(path_base + filename + ".wav", path_converted_audio + filename + ".wav")
            print(file + " is converted into " + filename +".wav")
        else:
            print("ERROR: Unsupported file type - "+ file + " was not converted. Modify the pre-processing stage to convert *" + file_extension + " files.")


def resample(file, sr):
    print("\nResampling of " + file + " in progress")
    path = path_converted_audio + file
    audio, sr = librosa.load(path, sr=sr)
    length = librosa.get_duration(audio, sr)

    print("File " + file + " is",datetime.timedelta(seconds=round(length,0)),"sec. long")
    sf.write(os.path.join(resampled_folder,file), audio, sr)
    resampled_path = os.path.join(resampled_folder,file)

    print(file + " was resampled to " + str(sr) + "kHz")
    return resampled_path, length


def asr_transcript(processor, model, resampled_path, length, block_length):
    chunks = length//block_length
    if length%block_length != 0:
        chunks += 1
    transcript = ""

    stream = librosa.stream(resampled_path, block_length=block_length, frame_length=16000, hop_length=16000)

    print ('Every chunk is ',block_length,'sec. long')
    print("Total number of chunks:",int(chunks))

    for n, speech in enumerate(stream):
        print ("Transcribing chunk number " + str(n+1))
        separator = ' '
        if n % 2 == 0:
            separator = '\n'
        transcript += generate_transcription(speech, processor, model) + separator
    print("Encoding complete. Total number of chunks: " + str(n+1) + "\n")
    return transcript


def generate_transcription(speech, processor, model):
    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]
    input_values = processor(speech, sampling_rate = sr, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()


def text_analysis(transcript, language, audio_report, file, length):
    print("\nProcessing " + file + " with NLU.")
    client = ExpertAiClient()
    output = client.specific_resource_analysis(body={"document": {"text": transcript}},
             params={'language': language, 'resource': 'relevants'})

    today = date.today()
    report = f"REPORT\nFile name: {file}\nDate: {today}" \
             f"\nLength: {datetime.timedelta(seconds=round(length,0))}" \
             f"\nFile stored at: {os.path.join(audio_report, file.split('.')[0])}.txt"

    report += "\n\nMAIN LEMMAS:\n"
    for lemma in output.main_lemmas:
        report += lemma.value + "\n"
    report += "\nMAIN PHRASES:\n"
    for lemma in output.main_phrases:
        report += lemma.value + "\n"
    report += '\nMAIN TOPICS:\n'
    for n,topic in enumerate(output.topics):
        if topic.winner:
            report += '#' + topic.label + '\n'

    filepath = os.path.join(audio_report,file)
    text = open(filepath[:-4] + ".txt","w")
    text.write(report)
    text.close()
    print("\nReport stored at " + filepath[:-4] + ".txt")
    return report


def speech_to_data(uploaded_file):
    preprocessing(path_base, path_converted_audio)
    model, processor = instantiate_model()

    for file in os.listdir(path_converted_audio):
        resampled_path, length = resample(file, sr)
        print("\nTranscribing ", file)
        transcript = asr_transcript(processor, model, resampled_path, length, block_length)
        print(transcript)
        with open(os.path.join(transcripts, uploaded_file),"w") as f:
            f.write(str(transcript))
        report = text_analysis(transcript, language, audio_report, file, length)
    shutil.rmtree(path_converted_audio)

clean_directory(path_base)
st.image(main_image,use_column_width='auto')
st.title("🗣 Speech <2> Data 🚀")
st.info('✨ Supports all popular audio formats - MP3, MP4, WAV, M4A, FLAC, OPUS 😉')
uploaded_file = st.file_uploader("Upload audio file", type=['mp3','mp4','m4a','flac','opus', 'wav'])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    with open(os.path.join(path_base,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
    st.markdown("Feel free to play your uploaded audio file 🎼")
    st.audio(audio_bytes)

    if "load_state" not in st.session_state:
        st.session_state.load_state = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Transcript") or st.session_state.load_state:
            st.session_state.load_state = True
            with st.spinner(f"Working... 💫 This make take several minutes depending on the file size."):
                speech_to_data(str(uploaded_file.name.split('.')[0]+".txt"))

            output_transcript_file = str(uploaded_file.name.split('.')[0]+".txt")
            output_file = open(os.path.join(transcripts, output_transcript_file),"r")
            output_transcript_data = output_file.read()

            download_transcript = st.download_button(
                                 label="Download Transcript 📝",
                                 data=output_transcript_data,
                                 file_name=output_transcript_file,
                                 mime='text/plain'
                         )
            if download_transcript:
                st.balloons()
                st.success('✅ Download Successful !!')
            with col2:
                if st.checkbox("Generate Text Analysis Report"):
                    output_report_file = str(uploaded_file.name.split('.')[0]+".txt")
                    output_file = open(os.path.join(audio_report, output_report_file),"r")
                    output_report_data = output_file.read()

                    download_report = st.download_button(
                                         label="Download Text Analysis Report 📝",
                                         data=output_report_data,
                                         file_name=output_report_file,
                                         mime='text/plain'
                                 )
                    if download_report:
                        st.balloons()
                        st.success('✅ Download Successful !!')
    clean_directory(path_base)

else:
    st.warning('⚠ Please upload your audio file 😯')

st.markdown("<br><hr><center>Made with ❤️ by <a href='mailto:ralhanprateek@gmail.com?subject=Speech 2 Data WebApp!&body=Please specify the issue you are facing with the app.'><strong>Prateek Ralhan</strong></a> with the help of [speech-2-data](https://github.com/therealexpertai/speech-2-data) built by [therealexpertai](https://github.com/therealexpertai) ✨</center><hr>", unsafe_allow_html=True)
st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)
