from __antennahandler import create_antenna_thread
from __audiohandler import create_audio_thread
from __keyboardthread import KeyboardThread, merge_inputs
from typing import Optional
from datetime import datetime
import os
import time
import re

DATA_FOLDER = "input_files"
# usable when using earphone for labeling
LAB_AUDIO_LABELED_DIR = "lab_audio_labeled"


def dir_handler(path: str, parent_path: Optional[str] = None, path_is_dir: bool = True) -> str:
    if parent_path is not None:
        if not os.path.isdir(parent_path):
            os.mkdir(parent_path)
        path = os.path.join(parent_path, path)
    if path_is_dir:
        os.mkdir(path)
    return path


def create_lab_train_set_without_break(dir_path: str, using_antenna: bool):
    """
    creating a folder named "dir_path" containing 4 files:
    a csv file for audio, a csv file for the serial reading, a file contain all the recognized text for debugging
    and a csv file containing the merged input.
    the input phase is stopped with saying the stop words in the AudioHandler
    :param dir_path: directory path name
    :param using_antenna: True if using antenna, false otherwise (using arduino instead)
    :return: None
    """
    dir_handler(dir_path)
    audio_thread = create_audio_thread(dir_path, recognize_while_streaming=True)
    antenna_thread = create_antenna_thread(dir_path, timestamp_working=using_antenna)
    audio_thread.start()
    antenna_thread.start()
    audio_thread.join()
    print("audio thread had stopped")
    antenna_thread.stop()
    merge_inputs(dir_path=dir_path)


def create_lab_train_set_with_break(dir_path: Optional[str] = None, timestamp_working: Optional[bool] = True):
    """
    creating a folder named "dir_path" containing 4 files:
    a csv file for audio, a csv file for the serial reading, a file contain all the recognized text for debugging
    and a csv file containing the merged input.
    This function is using the KeyboardThread class for manually changing the state of the program with a press
    :param timestamp_working:
    :param dir_path: directory path name, if not given will create dir named AL_date (Audio Labeled)
    :return: None
    """
    if dir_path is None:
        dir_path = re.sub(r"\s|-|:", "_", str(datetime.now())).split(".")[0]
    dir_path = dir_handler(dir_path, parent_path=LAB_AUDIO_LABELED_DIR)
    keyboard_thread = KeyboardThread(dir_path, timestamp_working=timestamp_working)
    keyboard_thread.start()
    print("all is finished!")


def create_train_set_from_serial_only(dir_path: Optional[str] = None, timestamp_working: Optional[bool] = True):
    if dir_path is None:
        dir_path = re.sub(r"\s|-|:", "_", str(datetime.now())).split(".")[0]
    dir_path = dir_handler(dir_path, parent_path=DATA_FOLDER, path_is_dir=False)
    antenna_thread = create_antenna_thread(dir_path, timestamp_working=timestamp_working, use_exact_dir_name=True)
    antenna_thread.start()
    return antenna_thread


def print_serial():
    temp_dir = f"temp_{int(time.time())}"
    a_h = create_antenna_thread(temp_dir).antenna_handler
    a_h.print_pretty_serial()
    os.rmdir(temp_dir)


def main():
    antenna = create_train_set_from_serial_only()
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        antenna.stop()
        print("stopped recording")


if __name__ == '__main__':
    main()
    # python -m serial.tools.list_ports
