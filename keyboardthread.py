from antennahandler import create_antenna_thread, ANTENNA_PATH
from audiohandler import create_audio_thread, AUDIO_PATH
import time
import re
import os
import keyboard
import threading

MERGE_PATH = "merged.csv"


class KeyboardThread(threading.Thread):
    """
    A class for running a thread which can change it states with a keyboard press
    """
    STARTING_STATE = "starting state"
    LISTENING_STATE = "listening state"
    WAITING_STATE = "waiting state"
    MERGING_STATE = "merging state"
    FINISHED_STATE = "finish state"
    STATE_HOT_KEY = 'ctrl+p'

    def __init__(self, dir_path: str, timestamp_working: bool = True):
        super().__init__()
        self.dir_path = dir_path
        self.state = KeyboardThread.STARTING_STATE
        self.antenna_thread = create_antenna_thread(dir_path=dir_path, timestamp_working=timestamp_working)
        self.audio_thread = create_audio_thread(dir_path=dir_path, recognize_while_streaming=False)
        self.set_keyboard_hotkeys()

    def change_state(self):
        """
        every time that called change the state of self
        :return:
        """
        print(f'changing state from {self.state}.')
        if self.state == KeyboardThread.STARTING_STATE:
            # start to listen
            self.state = KeyboardThread.LISTENING_STATE
            self.antenna_thread.start()
            self.audio_thread.start()

        elif self.state == KeyboardThread.LISTENING_STATE:
            # stop listening, recognize audio and wait for changes
            self.state = KeyboardThread.WAITING_STATE
            self.antenna_thread.stop()
            self.audio_thread.stop()
            # recognize all audio
            time.sleep(5.5)  # sometimes the audio thread still saved the listened audio
            self.audio_thread.recognize_saved_audio()
            # here we can refresh the project directory and see the new files, editing them if there are problems.
            # when finished we will again manually change self.state
            print("\t\t***\n"
                  "the audio files are ready and can be changed manually if needed.\n"
                  f"the code will continue after pressing: {KeyboardThread.STATE_HOT_KEY} again.\n"
                  f"\t\t***")

        elif self.state == KeyboardThread.WAITING_STATE:
            # stop waiting, start merging the files for final training set
            self.state = KeyboardThread.MERGING_STATE
            merge_inputs(self.dir_path)
            self.state = KeyboardThread.FINISHED_STATE

        else:
            print("wanted state change while in finished state")

        print(f'state is now {self.state}.')

    def set_keyboard_hotkeys(self):
        """
        set the pressing event and what to do with each press
        :return: None
        """
        print('Setting keyboard hotkeys...')
        keyboard.add_hotkey(KeyboardThread.STATE_HOT_KEY, self.change_state)

    def start(self):
        """
        start the thread
        :return:
        """
        print("keyboard_thread is ready to start!")
        while self.state != KeyboardThread.FINISHED_STATE:
            keyboard.wait(hotkey=KeyboardThread.STATE_HOT_KEY)
        print("KeyboardHandler had finished running")


def merge_inputs(dir_path: str):
    """
    while reading the serial input and the audio labels,
    maintaining a dict for all the id's statuses (moving, active, location)
    and creating a final csv file- which is the train set
    :param dir_path: the dir to which takes the inputs
    :return: None
    """
    def format_epc(epc: str) -> str:
        epc = str(epc).zfill(4)
        epc = "3" + "3".join(epc)
        return epc

    def update_epc_status():
        audio_dict.pop("Time")
        epc = audio_dict.pop("EPC")
        prev_status = epc_status.get(epc, epc_starting_labels)
        values_dict = dict()
        for key, value in audio_dict.items():
            if value == "-":
                value = prev_status.get(key, "0")
            values_dict[key] = value
        epc_status.update({epc: values_dict})

    def write_to_merged_file():
        epc_values_dict = epc_status.get(antenna_dict["EPC"], epc_starting_labels)
        labels_values = ",".join(epc_values_dict.values())
        antenna_values = ",".join(antenna_dict.values())
        merged_file.write(f"{antenna_values},{labels_values}\n")

    def next_line(file_name) -> str:
        current_line = next(file_name)
        return re.sub(r"[\s\n]+", "", current_line)

    def process_audio_line():
        try:
            line_ = next_line(audio_file)
            dict_ = dict(zip(audio_columns, line_.split(",")))
            dict_["EPC"] = format_epc(dict_["EPC"])
        except StopIteration:
            line_ = "_"
            dict_ = {"Time": float("inf")}
        return line_, dict_

    def process_antenna_line():
        line_ = next_line(antenna_file)
        dict_ = dict(zip(antenna_columns, line_.split(",")))
        return line_, dict_

    audio_file = open(os.path.join(dir_path, AUDIO_PATH), 'r')
    audio_columns = [column_name for column_name in next_line(audio_file).split(",")]
    antenna_file = open(os.path.join(dir_path, ANTENNA_PATH), 'r')
    antenna_columns = [column_name for column_name in next_line(antenna_file).split(",")]
    merged_path = os.path.join(dir_path, MERGE_PATH)
    merged_header = antenna_columns.copy()
    epc_starting_labels = {}
    epc_status = dict()
    for label in audio_columns:
        if label not in ["EPC", "Time"]:
            epc_starting_labels[label] = "0"
            merged_header.append(label)

    if not os.path.exists(merged_path):
        with open(fr"{merged_path}", 'x') as file:
            file.write(",".join(merged_header) + "\n")

    audio_line, audio_dict = process_audio_line()
    antenna_line, antenna_dict = process_antenna_line()
    with open(merged_path, 'a', encoding="utf-8") as merged_file:
        try:
            while True:
                if float(audio_dict["Time"]) <= float(antenna_dict["Time"]):
                    update_epc_status()
                    audio_line, audio_dict = process_audio_line()
                else:
                    write_to_merged_file()
                    antenna_line, antenna_dict = process_antenna_line()
        except StopIteration:
            print('the finished file is done!')
