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

# Choose TTS model, you can see more models in by print(TTS.list_models())
model_name = "tts_models/en/ljspeech/tacotron2-DDC"
# Init TTS
tts_model = TTS(model_name)
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
llm_model = Llama(model_path="G:/oobabooga_windows/text-generation-webui/models/vicuna-13b-v1.3.0.ggmlv3.q5_K_M.bin")

def text_to_speech_file(input_text):
    try:
        tts_model.tts_to_file(text=input_text, file_path="temp/agent_sentence.wav")
        return "temp/agent_sentence.wav"
    except Exception as e:
        print("An error occurred during text to speech conversion:", str(e))
        return None

def play_wav(file_path):
    chunk = 1024

    # Open the WAV file
    wf = wave.open(file_path, 'rb')

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

def transcribe_audio(file_path):
    segments, info = whisper_model.transcribe(file_path, beam_size=5)

    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "

    return transcribed_text.strip()

def get_response_from_agent(input_text):
    output = llm_model(f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {input_text}
ASSISTANT:""", max_tokens=100, stop=["USER:"], echo=False)
    output_text = output['choices'][0]['text']
    return output_text

print(get_response_from_agent("Can you give me a frog leg recipe from don't starve together?"))