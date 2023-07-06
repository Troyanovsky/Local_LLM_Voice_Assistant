# pip install TTS
# pip install wave pyaudio
# pip install faster-whisper
# pip install llama-cpp-python
from TTS.api import TTS
import pyaudio
import wave
from faster_whisper import WhisperModel
from llama_cpp import Llama
import json
import sounddevice as sd
import os

def delete_temporary_files(temp_folder):
    for file_name in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".wav"):
            os.remove(file_path)

def text_to_speech_file(input_text, tts_model):
    try:
        tts_model.tts_to_file(text=input_text, file_path="temp/agent_sentence.wav")
        return "temp/agent_sentence.wav"
    except Exception as e:
        print("An error occurred during text to speech conversion:", str(e))
        return None

def play_wav(file_path):
    chunk = 1024

    # Open the WAV file
    with wave.open(file_path, 'rb') as wf:
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Open the audio stream
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # Read data in chunks and play the audio
        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)

        # Close the stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

def transcribe_audio(file_path, whisper_model):
    segments, info = whisper_model.transcribe(file_path, beam_size=5)

    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "

    return transcribed_text.strip()

def get_response_from_agent(input_text, llm_model):
    output = llm_model(f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {input_text}
ASSISTANT:""", max_tokens=100, stop=["USER:"], echo=False)
    output_text = output['choices'][0]['text']
    return output_text

def main_loop():
    # Set up the temporary folder path for storing the audio files
    temp_folder = "./temp"
    os.makedirs(temp_folder, exist_ok=True)

    # Choose TTS model, you can see more models in by print(TTS.list_models())
    model_name = "tts_models/en/ljspeech/tacotron2-DDC"
    # Init TTS
    tts_model = TTS(model_name)
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    llm_model = Llama(model_path="G:/oobabooga_windows/text-generation-webui/models/vicuna-13b-v1.3.0.ggmlv3.q5_K_M.bin")

    while True:

        # Need to implement: record user voice

        user_input = transcribe_audio(user_wav_path, whisper_model)

        agent_output = get_response_from_agent(user_input, llm_model)

        agent_wav_path = text_to_speech_file(agent_output, tts_model)

        play_wav(agent_wav_path)

        delete_temporary_files(temp_folder)


if __name__ == "__main__":
    main_loop()