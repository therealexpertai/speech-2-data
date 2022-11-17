# 🗣 Speech <2> Data Streamlit based Web App ✨ [![](https://img.shields.io/badge/Prateek-Ralhan-brightgreen.svg?colorB=ff0000)](https://prateekralhan.github.io/)
A minimalistic automatic speech recognition streamlit based webapp powered by wav2vec2-base-960h Facebook model provided by HuggingFace transformers and the NL API provided by expert.ai

![demo](https://user-images.githubusercontent.com/29462447/202569539-db8b821e-352b-4e68-a364-8c737f46c524.gif)


## Installation:
* Simply run the command ***pip install -r requirements.txt*** to install the necessary dependencies.
* Get your API usage Credentials for NL API from expert.ai. You can register [here](https://developer.expert.ai/ui/login) for free tier access.

## Usage:
1. Enter your credentials in the ```creds.py``` file. You can also save them as your system environment variables as per your convenience.
2. Simply run the command: 
```
streamlit run app.py
```
3. Navigate to http://localhost:8501 in your web-browser. This will launch the web app :

![1](https://user-images.githubusercontent.com/29462447/202569353-cf3fc9ca-802d-4b67-a8b5-91304419932e.png)
![2](https://user-images.githubusercontent.com/29462447/202569360-dba1ac45-58fb-489e-9e09-dc0841ab1f28.png)

4. By default, streamlit allows us to upload files of **max. 200MB**. If you want to have more size for uploading audio files, execute the command :
```
streamlit run app.py --server.maxUploadSize=1028
```

### Running the Dockerized App
1. Ensure you have Docker Installed and Setup in your OS (Windows/Mac/Linux). For detailed Instructions, please refer [this.](https://docs.docker.com/engine/install/)
2. Navigate to the folder where you have cloned this repository ( where the ***Dockerfile*** is present ).
3. Build the Docker Image (don't forget the dot!! :smile: ): 
```
docker build -f Dockerfile -t app:latest .
```
4. Run the docker:
```
docker run -p 8501:8501 app:latest
```

This will launch the dockerized app. Navigate to ***http://localhost:8501/*** in your browser to have a look at your application. You can check the status of your all available running dockers by:
```
docker ps
```


