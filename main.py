from antennahandler import create_antenna_thread
from audiohandler import create_audio_thread
from keyboardthread import KeyboardThread, merge_inputs
from model import Model
import os
import time
from antennahandler import AntennaHandler
from datetime import datetime


def create_train_set_without_break(dir_path: str, using_antenna: bool):
    """
    creating a folder named "dir_path" containing 4 files:
    a csv file for audio, a csv file for the serial reading, a file contain all the recognized text for debugging
    and a csv file containing the merged input.
    the input phase is stopped with saying the stop words in the AudioHandler
    :param dir_path: directory path name
    :param using_antenna: True if using antenna, false otherwise (using arduino instead)
    :return: None
    """
    os.mkdir(dir_path)
    audio_thread = create_audio_thread(dir_path, recognize_while_streaming=True)
    antenna_thread = create_antenna_thread(dir_path, timestamp_working=using_antenna)
    audio_thread.start()
    antenna_thread.start()
    audio_thread.join()
    print("audio thread had stopped")
    antenna_thread.stop()
    merge_inputs(dir_path=dir_path)


def create_train_set_with_break(dir_path: str, timestamp_working: bool = True):
    """
    creating a folder named "dir_path" containing 4 files:
    a csv file for audio, a csv file for the serial reading, a file contain all the recognized text for debugging
    and a csv file containing the merged input.
    This function is using the KeyboardThread class for manually changing the state of the program with a press
    :param dir_path: directory path name
    :param using_antenna: True if using antenna, false otherwise (using arduino instead)
    :return: None
    """
    os.mkdir(dir_path)
    keyboard_thread = KeyboardThread(dir_path, timestamp_working=timestamp_working)
    keyboard_thread.start()
    print("all is finished!")


def main():
    # dir_path = f"test_sets\\test_{int(time.time())}"
    # create_train_set_with_break(dir_path, timestamp_working=True)

    # dir_path = "test_sets\\test_1632923100"
    # merge_inputs(dir_path)

    dir_path = f"test_sets\\test_{int(time.time())}"
    a_h = create_antenna_thread(dir_path).antenna_handler
    a_h.print_serial_port(True)

    # path = "_test_dummy/merged.csv"
    # model = Model(uni_model_path="model.eh")
    # # model.train(path)
    # model.predict_stream(port=AntennaHandler.find_arduino_device())


if __name__ == '__main__':
    main()
    # python -m serial.tools.list_ports
