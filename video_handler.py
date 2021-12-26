import cv2
from pathlib import Path
import math
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from preprocess import plot_antenna_df, check_de_sync_groups, plot_rssi_file, plot_rssi_df
from typing import Optional, Tuple
import os
import ffmpeg


def create_rfid_index():
    df = pd.read_csv("helper files/RFID_item_tag_index.csv", header=0)
    df.columns = [0, 1, 2, 3, 4, 5, 6, 7]
    df = df[[0, 3, 4, 5, 6]]
    df.dropna(inplace=True)
    df = df.astype({0: 'str', 3: 'int', 4: 'int', 5: 'int', 6: 'int'}).astype('str')
    df[8] = df[6] + df[5] + df[4] + df[3]
    df = df[[0, 8]]
    df.columns = ["name", "epc"]
    df = df[~df.name.str.contains("Cone Syringe")]

    syringes_name = ['Cone Syringe (50 cc)', 'Cone Syringe (20 cc)', 'Cone Syringe (10 cc)', 'Cone Syringe (5 cc)']
    syringes_epc = ['30303036', '30303037', '30303038', '30303039']
    names = []
    epcs = []
    for name, epc in zip(syringes_name, syringes_epc):
        for i in range(10):
            relevant_digit = epc[-1]
            new_epc = f"30303{relevant_digit}3{i}"
            names.append(name)
            epcs.append(new_epc)
    syringes_df = pd.DataFrame(list(zip(names, epcs)), columns=['name', 'epc'])
    df = pd.concat([df, syringes_df], ignore_index=True)
    df.sort_values(by=['epc'], inplace=True)
    df.to_csv("helper files/rfid_index.csv", index=False)
    print("a")


def show_video(video_path, video_name):
    video = cv2.VideoCapture(video_path)
    fps = math.floor(video.get(cv2.CAP_PROP_FPS))
    font = cv2.FONT_HERSHEY_SIMPLEX
    # frames_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # seconds = frames_num/fps
    frame_delay = 1000 // fps  # millisecends
    # frame_delay = 1
    starting_timestamp = time.mktime(time.strptime('10-21-21_09-10-47', '%m-%d-%y_%H-%M-%S'))
    # cv2.namedWindow(video_name)
    ret, frame = video.read()

    frame_counter = 0
    while video.isOpened():
        frame_counter += 1
        ret, frame = video.read()

        # frame = cv2.resize(frame, resolution)
        timestamp = starting_timestamp + frame_counter // fps
        date = datetime.fromtimestamp(timestamp)
        text_output = f"{date}, {frame_counter=}"
        cv2.putText(frame, text_output, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
        # waits for 1000/fps milliseconds time and check if the 'q' button is pressed
        # or video.read is finished (ret= false)
        if (cv2.waitKey(frame_delay) & 0xFF) == ord('q') or not ret:
            break
        cv2.imshow(video_name, frame)
    video.release()
    cv2.destroyAllWindows()


def show_plot(file):
    frame_delay = 20
    num_frames = 100

    fig, plot_ax = plt.subplots(nrows=1, ncols=1, figsize=(50, 30))
    plot_ax = preprocess_antenna_file(file_path=file, ax=plot_ax)
    xs = np.linspace(plot_ax.get_xlim()[0], plot_ax.get_xlim()[1], num_frames)
    line = plot_ax.axvline(xs[0])  # create vertical line in most left side position

    def animate(i):
        line.set_xdata(xs[i])  # update the line position on x axis
        return line,

    ani = animation.FuncAnimation(
        fig, animate, frames=num_frames, interval=frame_delay, blit=True, save_count=50)

    plt.show()


def preprocess_antenna_file(file_path, ax: plt.Axes, starting_time: Optional[float] = None,
                            ending_time: Optional[float] = None, antenna: Optional[int] = None,
                            computer_date: bool = True):
    df = pd.read_csv(filepath_or_buffer=file_path, dtype=str)
    rfid_index = pd.read_csv(filepath_or_buffer="helper files/rfid_index.csv", dtype=str).to_dict(orient='list')
    epc_2_name = dict(list(zip(rfid_index['epc'], rfid_index['name'])))
    df, problematic_indexes = check_de_sync_groups(df)
    df.drop(problematic_indexes, axis=0, inplace=True)
    columns = df.columns.values
    computer_column = [i for i in columns if 'omputer' in i]
    if len(computer_column) > 0:
        df.rename(columns={computer_column[0]: 'Computer_date'}, inplace=True)
    else:
        df['Computer_date'] = df['Date']
    df = df[["EPC", "Time", "Date", "Antenna", "RSSI", "Computer_date"]].astype(
        {'EPC': str, 'Time': 'int', 'Date': 'datetime64', 'Antenna': 'int', 'RSSI': 'int',
         'Computer_date': 'datetime64'})

    if not starting_time:
        starting_time = df["Time"].min()
    if not ending_time:
        ending_time = df["Time"].max()
    starting_date = datetime.fromtimestamp(starting_time)
    ending_date = datetime.fromtimestamp(ending_time)
    df = df[df["Time"].between(starting_time, ending_time)]
    if computer_date:
        df['Date'] = df['Computer_date']
    df.drop('Computer_date', axis=1, inplace=True)
    df["Tool_name"] = df["EPC"].map(epc_2_name)
    # ax = plot_antenna_df(df=df, show=False, ax=ax, min_freq=1, antenna=antenna)
    ax = plot_rssi_df(df=df, show=False, ax=ax, antenna=antenna)

    ax.set_xlim([starting_date, ending_date])
    return ax


def video_writer(major_video_path, minor_video_path, csv_path, output_video_name, computer_date, video_name):
    major_video_cap = cv2.VideoCapture(major_video_path)
    minor_video_cap = cv2.VideoCapture(minor_video_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps = major_video_cap.get(cv2.CAP_PROP_FPS)
    h = int(major_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(major_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(major_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    starting_timestamp = time.mktime(time.strptime(video_name.split(".")[0], '%m-%d-%y_%H-%M-%S'))
    ending_timestamp = starting_timestamp + duration
    if fps == 0:
        raise Exception(f"\nthe video could not be loaded\n")

    # plot figure
    fig, ax1 = plt.subplots(figsize=(19, 5))
    ax1 = preprocess_antenna_file(file_path=csv_path, ax=ax1, starting_time=starting_timestamp,
                                  ending_time=ending_timestamp, antenna=1, computer_date=computer_date)

    # output video creation
    out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    cur_frame_num = 0
    print("start creating edited video...")
    while major_video_cap.isOpened():
        cur_frame_num += 1
        if cur_frame_num % 100 == 0:
            print(f"\tcurrent frame = {cur_frame_num}")
        main_frame_exists, curr_frame_major = major_video_cap.read()
        zoom_frame_exists, curr_frame_minor = minor_video_cap.read()
        if main_frame_exists and zoom_frame_exists:
            # concat two camera angles
            curr_frame_major = cv2.resize(curr_frame_major, (w // 2, h // 2))
            curr_frame_minor = cv2.resize(curr_frame_minor, (w // 2, h // 2))
            curr_frame = cv2.hconcat([curr_frame_major, curr_frame_minor])

            # add timestamp to image
            timestamp = major_video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            now = datetime.fromtimestamp(starting_timestamp + timestamp)
            curr_frame = cv2.putText(curr_frame, str(now), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

            # add graph with line
            line1 = ax1.axvline(x=now, color="blue")
            plt.tight_layout()
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image_from_plot = np.pad(image_from_plot, pad_width=((0, 30), (0, 0), (0, 0)), constant_values=255)
            image_from_plot = cv2.resize(image_from_plot, (w, h // 2))
            image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
            line1.remove()

            # merge and save
            new_frame = cv2.vconcat([curr_frame, image_from_plot])
            out.write(new_frame)

        else:
            break

    major_video_cap.release()
    minor_video_cap.release()
    out.release()

    cv2.destroyAllWindows()
    print("finished video feed creation")


def create_audio(audio_input, video_input, output_name):
    audio = ffmpeg.input(audio_input).audio
    input_video = ffmpeg.input(video_input).video
    print("merging audio to video...")
    ffmpeg.concat(input_video, audio, v=1, a=1).output(output_name).run(overwrite_output=True)


def make_video_from(video_name: str, csv_name: str, station_name: str, composed_video_suffix: str = "_video_only.mp4",
                    full_video_suffix: str = "_full.mp4"):
    video_dir_path = f"videos_{station_name}_station/{video_name}"
    if station_name == "sapir":
        major_video_path = os.path.join(video_dir_path, f"CAM_IC_5/{video_name}.mp4")
        minor_video_path = os.path.join(video_dir_path, f"CAM_IC_1/{video_name}.mp4")
    elif station_name == "robert":
        major_video_path = os.path.join(video_dir_path, f"Axis3/{video_name}.mp4")
        minor_video_path = os.path.join(video_dir_path, f"Axis4/{video_name}.mp4")
    else:
        raise FileNotFoundError
    if not os.path.exists(major_video_path) or not os.path.exists(minor_video_path):
        print("video not found")
        raise FileNotFoundError
    if not os.path.exists(csv_name):
        print("csv not found")
        raise FileNotFoundError
    composed_video_name = video_name + composed_video_suffix
    video_writer(major_video_path=major_video_path, minor_video_path=minor_video_path, csv_path=csv_name,
                 output_video_name=composed_video_name, computer_date=True, video_name=video_name)
    output_video_name = video_name + full_video_suffix
    create_audio(audio_input=major_video_path, video_input=composed_video_name, output_name=output_video_name)


def main():
    video_name = '10-21-21_14-40-01.000'
    csv_name = "lab_unlabeled/sapir_bob_21_10/2021_10_21_14_43_05"
    make_video_from(video_name=video_name, csv_name=csv_name, station_name="sapir")


# video_name = '10-22-21_10-26-14'
# video_path = 'axis_bob/VIDEO_NAME'
# video_dir_path = video_path.replace("VIDEO_NAME", video_name)
# csv_name = 'lab_unlabeled/axis_bob_22_10/2021_10_22_10_25_01'
# suffix = "Date_video_only.mp4"
# video_writer(video_dir_path=video_dir_path, video_name=video_name, csv_path=csv_name, output_video_suffix=suffix,
#              computer_date=False)
# edited_video = video_name + suffix
# create_audio(audio_dir_path=video_dir_path, audio_name=video_name, video_input=edited_video,
#              output_name=video_name + "Date_full.mp4")

if __name__ == '__main__':
    # create_rfid_index()
    main()
