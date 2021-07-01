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


# Import Wav2Vec Model and Processor
model = "facebook/wav2vec2-base-960h"
print("Loading the model: ", model)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

#VARIABLES
path_base = "Speech1/"                      #The folder where the audios are stored
sr = 16000                                  #The sampling rate conversion
block_length = 30                            #The chunk length
language = "en"               
expertai_account = "your_expert.ai_email"    #your expert.ai email account
expertai_psw = "your_expert.ai_psw"                    #your expert.ai psw
os.environ["EAI_USERNAME"] = expertai_account
os.environ["EAI_PASSWORD"] = expertai_psw

#Folders and Path Creation
audio_report = "Audio_report"                                  #the folder name where your report are stored
path_converted_audio = "Converted_audio/"                      #the temporary folder name for converted audios
resampled_folder = "Resampled_audio/"                          #the resampled audios folder name
Path(audio_report).mkdir(parents = True, exist_ok = True)           #it creates the audio report folder
Path(path_converted_audio).mkdir(parents = True, exist_ok = True)   #it creates the converted audio folder
Path(resampled_folder).mkdir(parents = True, exist_ok = True)       #it creates the resampled folder

#Conversion List
extension_to_convert = ['.mp3','.mp4','.m4a','.flac','.opus']       #list of extension


#Preprocessing function
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
            print(file + " converted in " + filename +".wav")
        else:
            print(file + " is not converted. File extension is not supported. Add it on converter.")
            
#Resample Function
def resample(file, sr): 
    print("\nResampling of " + file + " started")
    path = path_converted_audio + file
    audio, sr = librosa.load(path, sr=sr)                           #load and file resampling
    length = librosa.get_duration(audio, sr)                        #file lenght
    print("File " + file + " is",datetime.timedelta(seconds=round(length,0)),"sec. long")
    sf.write(os.path.join(resampled_folder,file), audio, sr)        #(resampled_folder + file, audio, sr)
    resampled_path = os.path.join(resampled_folder,file)            #resampled_folder + file
    print(file + " resampled at " + str(sr) + "kHz")
    return resampled_path, length


#Transcript function
def asr_transcript(processor, model, resampled_path, length, block_length):
    chunks = length//block_length
    if length%block_length != 0:
        chunks += 1
    tok  = time.time()
    transcript = ""   
    # Stream over 30 seconds chunks rather than load the full file
    stream = librosa.stream(resampled_path, block_length=block_length, frame_length=16000, hop_length=16000)
    
    print ('Every chunk is ',block_length,'sec. long')
    print("Number of chunks",int(chunks))
    for n, speech in enumerate(stream):
        print ("Transcribing the chunk number " + str(n+1))
        separator = ' '
        if n % 2 == 0:
            separator = '\n'
        transcript += generate_transcription(speech, processor, model) + separator
    tik = time.time()
    print ("tempo trascorso per lo Speech to Text: ",tik-tok)
    print("Encoding complete. Total number of chunks: " + str(n+1) + "\n")
    return transcript

#Embedded Speech to Text Function
def generate_transcription(speech, processor, model):
    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]   
    input_values = processor(speech, sampling_rate = sr, return_tensors="pt").input_values       
    logits = model(input_values).logits             
    predicted_ids = torch.argmax(logits, dim=-1)       
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()



#NLU Analysis
def text_analysis(transcript, language, audio_report, file, length):
    #keyphrase extraction
    print("NLU analysis of " + file + " started.")
    client = ExpertAiClient()
    output = client.specific_resource_analysis(body={"document": {"text": transcript}}, 
             params={'language': language, 'resource': 'relevants'})
    
    today = date.today()
    report = f"REPORT\nFile name: {file}\nDate: {today}"          f"\nLength: {datetime.timedelta(seconds=round(length,0))}"          f"\nFile stored at: {os.path.join(audio_report, file)}.txt"
    
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
            
    #write report
    filepath = os.path.join(audio_report,file)
    text = open(filepath + ".txt","w")
    text.write(report)
    text.close()
    print("\nReport stored at " + filepath + ".txt")
    return report



def speech_to_data():
    
    preprocessing(path_base, path_converted_audio)

    for file in os.listdir(path_converted_audio):
        resampled_path, length = resample(file, sr) #samled_name
        print("\nStarting transcription:", file)
        transcript = asr_transcript(processor, model, resampled_path, length, block_length)
        print(transcript)
        report = text_analysis(transcript, language, audio_report, file, length)
    shutil.rmtree(path_converted_audio)



#Run the Speech To Data
speech_to_data()  


