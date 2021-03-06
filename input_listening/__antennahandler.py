import serial
from serial.tools.list_ports import comports
import numpy as np
import pandas as pd
from typing import List, Optional
import os
import time
import threading
import re

from datetime import datetime
import time

ANTENNA_PATH = "antenna_test.csv"
SERIAL_COLUMNS = ["EPC", "Time", "ReadCount", "RSSI", "Antenna", "Frequency", "Phase"]


class AntennaThread(threading.Thread):
    """
    A class for running serial reading(using SerialHandler) in a separate thread
    """

    def __init__(self, output_path: str, timestamp_working: bool, port: Optional[str] = None):
        super().__init__()
        # start_time = 0
        # if not timestamp_working:
        #     start_time = int(time.time())
        self.antenna_handler = AntennaHandler(output_path=output_path, port=port, timestamp_working=timestamp_working)

    def run(self) -> None:
        print(f"starting {self.name}, a {type(self)} thread. timestamp: {time.time()}")
        self.antenna_handler.save_to_file()
        print("antenna thread finished")

    def stop(self):
        self.antenna_handler.running = False

    def solve_time_desync(self):
        self.antenna_handler.solve_time_desync()

    def __repr__(self):
        return super().__repr__() + f". With antenna handler {self.antenna_handler!r}"


class AntennaHandler:
    # Columns will always start with EPC and then Time
    BAUDRATE = 115200
    NUM_INPUTS = len(SERIAL_COLUMNS)

    def __init__(self, output_path: str, timestamp_working: bool = False, port: Optional[str] = None):
        if port is None:
            port = AntennaHandler.find_arduino_device()
        self.output_path = output_path
        self.port = port
        self.running = True
        self.delay = 0
        if not timestamp_working:
            self.solve_time_desync()

    def save_to_file(self, buffer_size: int = 5):
        """
        creating and writing the file containing self.port's inputs
        :param buffer_size:
        :return:
        """
        ser = serial.Serial(port=self.port, baudrate=AntennaHandler.BAUDRATE)
        header = SERIAL_COLUMNS
        header.append("Date")
        header.append("Computer Date")
        path = self.output_path
        if not os.path.exists(path):
            with open(fr"{path}", 'x') as file:
                file.write(",".join(header) + "\n")

        while self.running:
            with open(fr"{path}", 'a') as file:
                output = ""
                i = 0
                while i < buffer_size:
                    try:
                        raw = ser.read_until()
                        computer_now = datetime.now()
                        inputs = AntennaHandler.parse_raw(raw, self.delay)
                        inputs.append(str(datetime.fromtimestamp(float(inputs[1]))))
                        inputs.append(str(computer_now))
                        output += ",".join(inputs) + "\n"
                        i += 1
                    except ValueError:
                        pass
                print(f"writing to {path}:\n {output}")
                file.write(output)

    def get_port_data(self, port: str):
        ser = serial.Serial(port=port, baudrate=AntennaHandler.BAUDRATE)
        while self.running:
            raw = ser.read_until()
            inputs = AntennaHandler.parse_raw(raw, self.delay)
            yield ",".join(inputs) + "\n"

    @staticmethod
    def stream_to_df(ser: serial.Serial, n_data: int = 5) -> pd.DataFrame:
        """
        creating a n_data rows size pandas.DataFrame containing the last inputs from the serial port.
        not used right now, maybe using it for entering data to the classifier
        :param ser:
        :param n_data:
        :return:
        """
        matrix = np.empty(shape=(AntennaHandler.NUM_INPUTS, n_data))
        i = 0
        while i < n_data:
            raw = ser.read_until()
            try:
                matrix[:, i] = AntennaHandler.parse_raw(raw)
                i += 1
            except ValueError:
                pass
        return pd.DataFrame({SERIAL_COLUMNS[i]: matrix[i, :] for i in range(AntennaHandler.NUM_INPUTS)})

    @staticmethod
    def parse_raw(raw: bytes, delay: int = 0) -> List[str]:
        """
        parses the raw bytes input into str values
        :param raw:
        :param delay:
        :return:
        """
        raw = raw.decode('utf-8')
        # raw like 'EPC:30313133,Time:4883789,ReadCount:1,RSSI:-25,Antenna:3,Frequency:912250,Phase:71,,,\r\n'
        if re.match(pattern=r"EPC", string=raw) is None:
            raise ValueError
        words = raw.split(",")[:AntennaHandler.NUM_INPUTS]
        words = [word.split(":")[1] for word in words]
        words[1] = str(int(words[1]) + delay)
        if len(words[0]) != 8:
            raise ValueError
        return words

    @staticmethod
    def find_arduino_device():
        """
        find the wanted device to link to, searching for an Arduino named device.
        if not found return the first one found
        :return: the port name of the found device
        """
        list_ports = comports()
        if len(list_ports) == 0:
            print("no serial port was connected. terminating.")
            exit()
        for list_port in list_ports:
            if "A0-0319BBBK06A2" == list_port.serial_number or "C0-1718BBBK05F3" == list_port.serial_number:
                print(f"found an arduino device {list_port.description}")
                return list_port.device

        return list_ports[0].device

    def print_serial_port(self, parse: bool = False):
        """
        print the given port as serial value indefinitely
        :param parse:
        :param port:
        :return:
        """
        print(f"connecting to port {self.port}")
        ser = serial.Serial(port=self.port, baudrate=AntennaHandler.BAUDRATE)
        print("connected")
        while True:
            try:
                raw = ser.read_until()
                now = int(time.time())
                print(raw)
                if parse:
                    raw = self.parse_raw(raw=raw, delay=self.delay)
                    input_timestamp = int(raw[1])
                else:
                    raw = raw.decode('utf-8')
                    input_timestamp = int(raw.split(',')[1].split(':')[1])
                print(f"input timestamp {input_timestamp}, {datetime.fromtimestamp(input_timestamp)}\n")
                print(f"computer timestamp {now}, {datetime.fromtimestamp(int(now))}\n")

            except ValueError:
                pass

    def print_pretty_serial(self):
        print(f"connecting to port {self.port}")
        ser = serial.Serial(port=self.port, baudrate=AntennaHandler.BAUDRATE)
        print("connected")
        print("EPC, timestamp, date, RSSI, antenna, phase")
        while True:
            try:
                raw = ser.read_until()
                raw = self.parse_raw(raw=raw, delay=self.delay)
                raw = [raw[0], raw[1], str(datetime.fromtimestamp(float(raw[1]))), raw[3], raw[4], raw[-1]]
                print(raw)
            except ValueError:
                pass

    def solve_time_desync(self):
        print("solving desync...")
        ser = serial.Serial(port=self.port, baudrate=AntennaHandler.BAUDRATE)
        time_deltas = []
        while len(time_deltas) < 5:
            try:
                approx_time = int(time.time())
                raw = ser.read_until()
                raw = self.parse_raw(raw=raw, delay=self.delay)
                input_timestamp = int(raw[1])
                time_deltas.append(approx_time - input_timestamp)
            except ValueError:
                pass
        time_delta = round(sum(time_deltas) / len(time_deltas))
        print(f"found a {time_delta} sec delay")
        self.delay = time_delta


def create_antenna_thread(dir_path: str = None, timestamp_working: bool = True,
                          use_exact_dir_name: bool = False) -> AntennaThread:
    """
    A function for creating a serial reading thread with the right starting values
    :param use_exact_dir_name: determine if adding ANTENNA_PATH to output file path
    :param dir_path:
    :param timestamp_working: used for determine if to use the serial reader timestamp or to use the computer timestamp
    while using arduino device we want to use our computer timestamp
    due to the arduino's timestamp is zero in the start of the program
    :return: the created AntennaThread
    """
    path = ANTENNA_PATH
    if dir_path is not None:
        path = dir_path
        if not use_exact_dir_name:
            path = os.path.join(dir_path, ANTENNA_PATH)
    antenna_thread = AntennaThread(output_path=path, timestamp_working=timestamp_working)
    antenna_thread.daemon = True
    return antenna_thread
