import re
import time
from typing import Optional
import speech_recognition as sr
import os
import threading
from collections import OrderedDict
# tomer 1
AUDIO_PATH = "audio_test.csv"


class AudioThread(threading.Thread):
    """
    A class for running audio reading in a separated thread
    """

    def __init__(self, output_file: str, language: str = 'he', recognize_while_streaming: bool = True,
                 mic_name: Optional[str] = None):
        super().__init__()
        self.recognize_while_streaming = recognize_while_streaming
        self.audio_handler = AudioHandler(output_path=output_file, language=language, mic=mic_name)

    def run(self) -> None:
        print(f"starting {self.name}, a {type(self)} thread. timestamp: {time.time()}")
        if self.recognize_while_streaming:
            self.audio_handler.save_commands_to_file()
        else:
            self.audio_handler.save_audio_to_self()
        print("audio thread finished")

    def stop(self):
        self.audio_handler.running = False

    def recognize_saved_audio(self):
        self.audio_handler.recognized_saved_audio()

    def __repr__(self):
        return super().__repr__() + f". With audio handler {self.audio_handler!r}"


class AudioHandler:
    COLUMNS = ['EPC', 'Time', 'location', 'moving', 'active']
    SILENT_INPUT = "┬-silent-┬"
    PROBLEMATIC_INPUT = "┬-problem-┬"
    TERMINATE_INPUT = "┬-terminating thread-┬\n"

    def __init__(self, output_path: str, language: str = "en-US", mic: Optional[str] = None):
        self.audio_blocks = OrderedDict()
        self.output_path = output_path
        self.r = sr.Recognizer()
        self.language = language
        self.running = True
        self.mic = sr.Microphone(device_index=AudioHandler.find_microphone(wanted_mic=mic))

    def __repr__(self):
        return f"{self.output_path=}, {self.language=}, {self.mic=}"

    def get_phrase_audio(self, timeout: int = 0.5, phrase_time_limit: int = 5) -> sr.AudioData:
        """
        listen to audio and returning it. can raise sr.WaitTimeoutError
        :param timeout: how much to wait for input (in sec)
        :param phrase_time_limit: how much time to wait before stopping to listen
        :return: a AudioData object containing the saved audio
        """
        with self.mic as source:
            print("say something!")
            audio = self.r.listen(source, phrase_time_limit=phrase_time_limit, timeout=timeout)
        return audio

    def get_phrase_input(self, timeout: int = 0.5, phrase_time_limit: int = 5) -> (str, str):
        """
        listen to audio and recognize it immediately afterwards
        :param timeout: how much to wait for input (in sec)
        :param phrase_time_limit: how much time to wait before stopping to listen
        :return: the text recognized from teh audio, the labels derived from the text
        """
        text_from_input = ""
        try:
            timestamp: int = int(time.time())
            audio = self.get_phrase_audio(timeout=timeout, phrase_time_limit=phrase_time_limit)
            text_from_input: str = self.r.recognize_google(audio, language=self.language)
            print(f"speech to text output is {text_from_input}")
            text_command_format = self.get_inputs_command(text_from_input, timestamp) + "\n"
            return text_from_input, text_command_format
        except sr.WaitTimeoutError:
            print("audio was silent")
            return text_from_input, AudioHandler.SILENT_INPUT
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return text_from_input, AudioHandler.PROBLEMATIC_INPUT

    def save_commands_to_file(self):
        """
        write the text and the labels to the output files
        :return: None
        """
        audio_commands_path = self.create_output_file_header()

        while self.running:
            with open(fr"{self.output_path}", 'a', encoding="utf-8") as audio_csv_file:
                with open(fr"{audio_commands_path}", 'a', encoding="utf-8") as audio_text_file:
                    text, input_command = self.get_phrase_input()
                    if input_command == AudioHandler.SILENT_INPUT:
                        continue
                    elif input_command == AudioHandler.PROBLEMATIC_INPUT:
                        audio_csv_file.write(f"-1, {int(time.time())}, problem, problem, problem\n")
                    elif input_command == AudioHandler.TERMINATE_INPUT:
                        audio_text_file.write(f"{int(time.time())}, terminating thread\n")
                        return
                    else:
                        print(f"text: {text}")
                        audio_text_file.write(f"{int(time.time())}, {text}\n")
                        audio_csv_file.write(input_command)

    def save_audio_to_self(self):
        """
        listen and save the audio for later recognition
        :return: None
        """
        while self.running:
            try:
                timestamp: int = int(time.time())
                audio = self.get_phrase_audio()
                self.audio_blocks[timestamp] = audio
            except sr.WaitTimeoutError:
                print("audio was silent")
                continue

    def create_output_file_header(self):
        """
        creating 2 files, one will contain the recognised text and the other the labels
        :return: the text_file path
        """
        header = AudioHandler.COLUMNS
        path = self.output_path

        dir_path, file_name, *_ = path.split('\\')
        audio_text_only_path = f"audio_only_{file_name}"
        audio_text_only_path = os.path.join(dir_path, audio_text_only_path)

        if not os.path.exists(path):
            with open(fr"{path}", 'x') as file:
                file.write(",".join(header) + "\n")
        if not os.path.exists(audio_text_only_path):
            with open(fr"{audio_text_only_path}", 'x') as file:
                file.write("audio commands" + "\n")
        return audio_text_only_path

    def recognized_saved_audio(self):
        """
        create and write the 2 output files with recognising the saved audio
        :return: None
        """
        audio_text_only_path = self.create_output_file_header()
        with open(fr"{audio_text_only_path}", 'a', encoding="utf-8") as audio_text_file:
            with open(fr"{self.output_path}", 'a', encoding="utf-8") as audio_csv_file:
                for timestamp, audio in self.audio_blocks.items():
                    try:
                        text_from_input: str = self.r.recognize_google(audio, language=self.language)
                        print(f"speech to text output is {text_from_input}")
                        text_command_format = self.get_inputs_command(text_from_input, timestamp) + "\n"
                        audio_csv_file.write(text_command_format)
                        audio_text_file.write(f"{int(time.time())}, {text_from_input}\n")
                    except sr.UnknownValueError:
                        print("Google Speech Recognition could not understand audio")
                    except sr.RequestError as e:
                        print("Could not request results from Google Speech Recognition service; {0}".format(e))
                    continue

    def get_inputs_command(self, text: str, timestamp: int) -> str:
        """
        create the label set from a given sentence and timestamp
        :param text:
        :param timestamp:
        :return:
        """
        if self.language == 'he':
            move_commands = ['בתנועה', 'הורם']
            stop_commands = ['במנוחה', 'הונח']
            area_commands = ['מעל', 'זז']
            activation_commands = ['אקטיבי', 'active']
            passive_commands = ['פאסיבי', 'פסיבי']
            terminate_words = ['עצור', 'צור']
            num_words = {
                "אחד": "1", "אחת": "1", "שתיים": "2", "שניים": "2", "שלוש": "3", "ארבע": "4", "חמש": "5", "שש":
                    "6", "שבע": "7", "שמונה": "8", "תשע": "9", "אפס": "0"}

        else:
            move_commands = []
            stop_commands = []
            area_commands = []
            activation_commands = []
            passive_commands = []
            terminate_words = ['stop']
            num_words = {"one": "1"}

        for key, value in num_words.items():
            text = text.replace(key, value)
        text = re.sub(r"[\s,.:;]+", "", text)

        if text in terminate_words:
            return AudioHandler.TERMINATE_INPUT[:-1]

        epc, *location = re.findall(r"\d+", text)
        command = re.sub(r"\d+", "", text)
        if len(location) == 0:
            location = '-'
        else:
            location = location[0]

        moving = '-'
        active = '-'

        if command in move_commands:
            moving = 1
        elif command in stop_commands:
            moving = 0
        elif command in area_commands:
            moving = 1
        elif command in activation_commands:
            active = 1
        elif command in passive_commands:
            active = 0
        else:
            # pretty ugly, but used for removing the unknown prefix to the next words after the command
            command = command[:-1]
            if command in move_commands:
                moving = 1
            elif command in stop_commands:
                moving = 0
            elif command in area_commands:
                moving = 1
            elif command in activation_commands:
                active = 1
            elif command in passive_commands:
                active = 0

        output = f"{epc}, {timestamp}, {location}, {moving}, {active}"
        return output

    @staticmethod
    def find_microphone(wanted_mic: Optional[str] = None):
        """
        find a microphone to link to
        :param wanted_mic: the name of the wanted mic
        :return: the index of the wanted mic
        """
        if wanted_mic is None:
            wanted_mic = 'Microphone Array (Realtek(R) Au'  # my computer default mic
        mic_list = sr.Microphone.list_microphone_names()
        assert len(mic_list) != 0
        for i, microphone in enumerate(mic_list):
            if wanted_mic in microphone:
                print(f"found microphone {wanted_mic}")
                return i
        return 1


def create_audio_thread(dir_path, recognize_while_streaming: bool = True) -> AudioThread:
    """
    creating an AudioThread with the right starting values
    :param dir_path:
    :param recognize_while_streaming:
    :return:
    """
    mic = 'Headset Microphone (Poly BT600)'
    path = AUDIO_PATH
    if dir_path is not None:
        path = os.path.join(dir_path, AUDIO_PATH)
    audio_thread = AudioThread(output_file=path, mic_name=mic, recognize_while_streaming=recognize_while_streaming)
    return audio_thread
