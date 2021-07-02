# speech-2-data
### Find the full article at insert_the_URL

This script transcribes a speech into a text, analyzes it and then it writes the report in a txt file.

SETTING THE STAGE

First thing the script does is importing all the necessary libraries and model, and setting the variables.
 
https://gist.github.com/SamuelAlgheriniAI/b3fcddb1c776229aed1869295b1841c9

https://gist.github.com/SamuelAlgheriniAI/47eafe84b1029e09c78aa45f85eef561

There are plenty of Wav2Vec models on HuggingFace. I chose the “base-960h” because it is a good compromise between quality and weight structure. Write the path for your audio files in “path_base”. Leave the variable sr at 16000 (this is the sampling rate). You can also choose a different block length, depending on your CPU and RAM capabilities: I set it at 30 (the unit is seconds). Insert your expert.ai developer portal email and password in their respective variables. Then write the folders names you prefer for the conversion, resampling and final report. The program will create those paths for you (mkdir). You can increase the extension_to_convert list adding more extensions, if necessary.

PREPROCESSING

I begin preprocessing the audios. The aim is to get a folder filled by only .wav files.
 
https://gist.github.com/SamuelAlgheriniAI/0c3abed928485e64fc4b227bb6516606

The preprocessing function iterates through the original folder where your audio files are stored. If the file has a “.wav” extension, then it sends the file to the “path_converted_audio” folder, otherwise it converts such file to a “.wav” extension first. Two things: 1) in order to make this conversion work you must have ffmpeg.exe installed in the same folder of your running script; 2) if your file has an extension that is not in the “extension_to_convert” list, then it will not be converted and the program goes to the next iteration (it will give you a warning that the file has not been converted). 
As the FOR cycle in the preprocessing function comes to an end, I have the “path_converted_audio” filled with all “.wav” files. I am now ready to start the process that generates the text report. It is composed by three functions: resample, asr_transcript (and its nested generate_transcription function) and text_analysis.

RESAMPLING
 
https://gist.github.com/SamuelAlgheriniAI/1f2918fd1b64cd3d4749f61cd1987649

The resample function, as the name says, resamples the audio. It takes the file and the sampling rate as arguments. For my purpose I am resampling it at 16kHz but if you want to use it with other models that accept or need a different sampling rate, just change the “sr” variable in the variable section (or pass it directly to the function), and you get your desired sampling rate conversion. Here the function (librosa.load) loads the file, resampling it, and also gets the length information back (librosa.get_duration). Lastly, it stores the resampled file in the resample_path folder. The function returns the resampled_path and length.

SPEECH TO TEXT

Now I can pass the resampled audio to the asr_transcript function.
 
https://gist.github.com/SamuelAlgheriniAI/316e835a747b1930c11ee08bf78e71ab

The asr_transcript function takes five arguments: processor and model have been imported at the beginning section of the script, block_length has been set in the variables section (I assigned a value of 30, that means 30 seconds), and the resampled_path and length are returned from the previous function (resampled). At the beginning of the function, I immediately calculate how many pieces the audio consists of and then I instantiate “transcript” as an empty string. Then I apply to the file the librosa.stream function that returns (in fixed-length buffers) a generator, on which I iterate over to produce blocks of audio. I send each block to the generate_transcription function, the proper speech-to-text module that takes the speech (that is the single block of audio I am iterating over), processor and model as arguments and returns the transcription. In these lines the program converts the input in a pytorch tensor, retrieves the logits (the prediction vector that a model generates), takes the argmax (a function that returns the index of the maximum values) and then decodes it. The final transcription is all capital letters. In absence of casing, an NLP service like expert.ai handles this ambiguity better if everything is lowercase, and therefore I apply that case conversion.
So, when I call the asr_transcript function it takes the audio, iterates over it providing each time a block of the audio to the generate_transcription function, which in turn transcribes it and then appends this transcription to the previous one (creating a new line every two blocks).
At this point, I have got the transcription of our original audio file. It is time to analyze it.

TEXT ANALYSIS

It is information discovery time. Now that I have a transcript, I can query the expert.ai NL API service and generate the final report.

https://gist.github.com/SamuelAlgheriniAI/9c49bfac30eabf01da56c5a97b79649b

Text_analysis takes five arguments: transcript (returned from asr_transcript function), language and audio_report (already set in the variables section), file (it is the single file from the group I am iterating) and length (returned from the resample function). I instantiate the ExpertAiClient() calling it simply “client” and then I send my request. It is very simple, and it takes just one line of code. I specify the method (in my case “specific_resource_analysis”), and then I pass “transcript” as text, “language” as language and “relevants” as resource. This call is specific to my case, but with a slight modification you can query other types of analysis such as emotional traits, classification, NER, POS tagging, writeprint, PII and much more. Once I got the response back, I iterate through it extracting main lemmas, main phrases, and main topics, adding these responses to the report which is written in a .txt file stored in the audio_report folder.
We have done all the steps necessary to get data from an audio file. Finally, let’s look at the main function that executes all these other functions in the proper order.

SPEECH_TO_DATA

https://gist.github.com/SamuelAlgheriniAI/d31a98b62d715ca8e03895f67c06cc83

Speech_to_data is the function that drives the execution of the entire workflow. In other words, this is the one function we call to get data out of an audio file.
 
This function triggers the preprocessing function, that creates a folder with all converted files ready to be analyzed, and then iterates through every file. It resamples the file, then transcribes it, analyzes the text and generates the report. The last line of code removes the now useless path_converted_audio folder.

FINAL REMARKS

I enjoyed writing this code. Thanks to open source, Facebook AI, HuggingFace, and expert.ai, I have been able to get data from audio files just by using my home computer. And the list of potential applications I see is endless. 
