from __antennahandler import create_antenna_thread
from typing import Optional
from datetime import datetime
import os
import time
import re

DATA_FOLDER = "input_files"


def dir_handler(path: str, parent_path: Optional[str] = None, path_is_dir: bool = True) -> str:
    if parent_path is not None:
        if not os.path.isdir(parent_path):
            os.mkdir(parent_path)
        path = os.path.join(parent_path, path)
    if path_is_dir:
        os.mkdir(path)
    return path


def create_file_from_serial(dir_path: Optional[str] = None, timestamp_working: Optional[bool] = True):
    """
    create thread for listening to the serial port
    :param dir_path:
    :param timestamp_working:
    :return:
    """
    if dir_path is None:
        dir_path = re.sub(r"\s|-|:", "_", str(datetime.now()).split(".")[0]) + ".csv"
    dir_path = dir_handler(dir_path, parent_path=DATA_FOLDER, path_is_dir=False)
    antenna_thread = create_antenna_thread(dir_path, timestamp_working=timestamp_working, use_exact_dir_name=True)
    antenna_thread.start()
    return antenna_thread


def main():
    """
    creates a listening thread that saves all incoming input to a file, stops at key press
    :return:
    """
    antenna = create_file_from_serial()
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        antenna.stop()
        print("stopped recording")


if __name__ == '__main__':
    # shows in terminal all available ports, good for debugging: python -m serial.tools.list_ports
    main()
