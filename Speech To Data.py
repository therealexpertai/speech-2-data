#!/usr/bin/env python
# coding: utf-8


#import
import librosa
import torch
import time
import datetime
from pathlib import Path
import subprocess
import os
import shutil
import soundfile as sf  
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datetime import date
from expertai.nlapi.cloud.client import ExpertAiClient         #Expert.ai variables and import


# Import the Wav2Vec model and processor
model = "facebook/wav2vec2-base-960h"
print("Loading model: ", model)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

#VARIABLES
path_base = "Audio files/"                      #Original speech/audio files folder
sr = 16000                                  #Sampling rate
block_length = 30                            #Speech chunk size
language = "en"               
expertai_account = "your_expert.ai_email"    #Your expert.ai email account
expertai_psw = "your_expert.ai_psw"                    #Your expert.ai psw
os.environ["EAI_USERNAME"] = expertai_account
os.environ["EAI_PASSWORD"] = expertai_psw

#Folders and Path Creation
audio_report = "Reports"                                  #This is the folder where your report will be stored
path_converted_audio = "converted_files/"                      #This is the temporary folder for converted audio files
resampled_folder = "resampled_files/"                          #This is the folder for the resampled audio files
Path(audio_report).mkdir(parents = True, exist_ok = True)           #This creates the reports folder
Path(path_converted_audio).mkdir(parents = True, exist_ok = True)   #This creates the folder for converted audio files
Path(resampled_folder).mkdir(parents = True, exist_ok = True)       #This creates the folder for resampled audio files

#Conversion List
extension_to_convert = ['.mp3','.mp4','.m4a','.flac','.opus']       #List of the supported files types/extensions


#Pre-processing function
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
            
#Resampling function
def resample(file, sr): 
    print("\nResampling of " + file + " in progress")
    path = path_converted_audio + file
    audio, sr = librosa.load(path, sr=sr)                           #File load and resampling
    length = librosa.get_duration(audio, sr)                        #File lenght
    print("File " + file + " is",datetime.timedelta(seconds=round(length,0)),"sec. long")
    sf.write(os.path.join(resampled_folder,file), audio, sr)        #(resampled_folder + file, audio, sr)
    resampled_path = os.path.join(resampled_folder,file)            #resampled_folder + file
    print(file + " was resampled to " + str(sr) + "kHz")
    return resampled_path, length



#Transcription function
def asr_transcript(processor, model, resampled_path, length, block_length):
    chunks = length//block_length
    if length%block_length != 0:
        chunks += 1
    transcript = ""   
    # Split the speech in multiple 30 seconds chunks rather than loading the full audio file
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

#Speech to text function
def generate_transcription(speech, processor, model):
    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]   
    input_values = processor(speech, sampling_rate = sr, return_tensors="pt").input_values       
    logits = model(input_values).logits             
    predicted_ids = torch.argmax(logits, dim=-1)       
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()


#NLU analysis
def text_analysis(transcript, language, audio_report, file, length):
    #Keyphrase extraction
    print("\nProcessing " + file + " with NLU.")
    client = ExpertAiClient()
    output = client.specific_resource_analysis(body={"document": {"text": transcript}}, 
             params={'language': language, 'resource': 'relevants'})
    
    today = date.today()
    report = f"REPORT\nFile name: {file}\nDate: {today}" \
             f"\nLength: {datetime.timedelta(seconds=round(length,0))}" \
             f"\nFile stored at: {os.path.join(audio_report, file)}.txt"
    
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
            
    #Write the report to a text file
    filepath = os.path.join(audio_report,file)
    text = open(filepath + ".txt","w")
    text.write(report)
    text.close()
    print("\nReport stored at " + filepath + ".txt")
    return report


def speech_to_data():
    
    preprocessing(path_base, path_converted_audio)

    for file in os.listdir(path_converted_audio):
        resampled_path, length = resample(file, sr) #sampled_name
        print("\nTranscribing ", file)
        transcript = asr_transcript(processor, model, resampled_path, length, block_length)
        print(transcript)
        report = text_analysis(transcript, language, audio_report, file, length)
    shutil.rmtree(path_converted_audio)


#Run Speech-2-Data module
speech_to_data()  



