import argparse

import cv2
import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from df_utils import filter_df, plot_line_rssi_df, check_de_sync_groups, TOOL_GROUPS_NUM
from typing import Optional, Iterable
import os
import ffmpeg

VIDEOS_OUTPUT_DIR = "output_videos"
CSV_INPUT_DIR = "input_files"


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


def preprocess_antenna_file(file_path, axes: Iterable[plt.Axes], starting_time: Optional[float] = None,
                            ending_time: Optional[float] = None, antenna: Optional[int] = None,
                            computer_date: bool = True):
    df = pd.read_csv(filepath_or_buffer=file_path, dtype=str)
    rfid_index = pd.read_csv(filepath_or_buffer="helper files/rfid_index.csv", dtype=str).to_dict(orient='list')
    epc_2_name = dict(list(zip(rfid_index['EPC'], rfid_index['Name'])))
    df, problematic_indexes = check_de_sync_groups(df)
    df.drop(problematic_indexes, axis=0, inplace=True)
    computer_column = [i for i in df.columns.values if 'omputer' in i]
    if len(computer_column) > 0:
        df.rename(columns={computer_column[0]: 'Computer_date'}, inplace=True)
    else:
        df['Computer_date'] = df['Date']
    df = df[["EPC", "Time", "Date", "Antenna", "RSSI", "Computer_date", "group"]].astype(
        {'EPC': str, 'Time': 'int', 'Date': 'datetime64', 'Antenna': 'int', 'RSSI': 'int',
         'Computer_date': 'datetime64', 'group': 'int'})

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
    df = filter_df(df=df, antenna=antenna)

    axes = plot_line_rssi_df(df=df, show=False, axes=axes, antenna=antenna)
    # ax = plot_rssi_df(df=df, show=False, ax=ax, antenna=antenna)
    for ax in axes:
        text = "csv file: " + file_path.split("/")[-1]
        ax.text(starting_date, ax.get_ylim()[0], file_path.split("/")[-1], fontsize=8, ha="left", va="bottom")
        ax.set_xlim([starting_date, ending_date])
    return axes


def video_writer(major_video_path, minor_video_path, csv_path, output_videos_name, computer_date, video_name,
                 video_cap=float("inf")):
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
    figs, axes = [], []
    for i in range(TOOL_GROUPS_NUM):
        fig, ax = plt.subplots(figsize=(19, 5))
        figs.append(fig)
        axes.append(ax)

    axes = preprocess_antenna_file(file_path=csv_path, axes=axes, starting_time=starting_timestamp,
                                   ending_time=ending_timestamp, antenna=1, computer_date=computer_date)

    # output video creation
    out_videos = []
    for i in range(TOOL_GROUPS_NUM):
        tool_group = axes[i].get_title().split("_")[-1]
        output_video_name = output_videos_name.replace(".mp4", f"_{tool_group}.mp4")
        out_videos.append(cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)))

    cur_frame_num = 0
    print("start creating edited video...")
    while major_video_cap.isOpened():
        cur_frame_num += 1
        if cur_frame_num % 100 == 0:
            print(f"\tcurrent frame = {cur_frame_num}")
        main_frame_exists, curr_frame_major = major_video_cap.read()
        zoom_frame_exists, curr_frame_minor = minor_video_cap.read()
        if main_frame_exists and zoom_frame_exists and cur_frame_num < video_cap:
            # concat two camera angles
            curr_frame_major = cv2.resize(curr_frame_major, (w // 2, h // 2))
            curr_frame_minor = cv2.resize(curr_frame_minor, (w // 2, h // 2))
            curr_frame = cv2.hconcat([curr_frame_major, curr_frame_minor])

            # add timestamp to image
            timestamp = major_video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            now = datetime.fromtimestamp(starting_timestamp + timestamp)
            curr_frame = cv2.putText(curr_frame, str(now), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

            for i in range(TOOL_GROUPS_NUM):
                # add graph with line
                fig = figs[i]
                ax = axes[i]
                line1 = ax.axvline(x=now, color="blue")
                fig.tight_layout()
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.pad(image_from_plot, pad_width=((0, 30), (0, 0), (0, 0)), constant_values=255)
                image_from_plot = cv2.resize(image_from_plot, (w, h // 2))
                image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
                line1.remove()

                # merge and save
                new_frame = cv2.vconcat([curr_frame, image_from_plot])
                out_videos[i].write(new_frame)

        else:
            break

    major_video_cap.release()
    minor_video_cap.release()
    for out in out_videos:
        out.release()

    cv2.destroyAllWindows()
    print("finished video feed creation")


def create_audio(audio_input, videos_input, output_suffix):
    audio = ffmpeg.input(audio_input).audio
    for video_name in os.listdir(videos_input):
        video_path = os.path.join(videos_input, video_name)
        input_video = ffmpeg.input(video_path).video
        print("merging audio to video...")
        output_name = video_path.replace(".mp4", f"{output_suffix}.mp4")
        ffmpeg.concat(input_video, audio, v=1, a=1).output(output_name).run(overwrite_output=True)


def make_video_from(video_name: str, csv_name: str, station_name: str, composed_video_suffix: str = "_video_only.mp4",
                    full_video_suffix: str = "_full.mp4", computer_date: bool = True, video_cap=float("inf")):
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
    output_dir_path = os.path.join(VIDEOS_OUTPUT_DIR, f"output_{video_name}")
    os.makedirs(output_dir_path)
    composed_video_name = video_name + composed_video_suffix
    composed_video_name = os.path.join(output_dir_path, composed_video_name)
    video_writer(major_video_path=major_video_path, minor_video_path=minor_video_path, csv_path=csv_name,
                 output_videos_name=composed_video_name, computer_date=computer_date, video_name=video_name,
                 video_cap=video_cap)

    create_audio(audio_input=major_video_path, videos_input=output_dir_path, output_suffix="_with_audio")


def find_closest(video_name, station) -> str:
    # TODO: fix
    sub_dir = f"{station}_{video_name.split('_')[0]}"
    dir_path = os.path.join(CSV_INPUT_DIR, sub_dir)
    wanted_timestamp = datetime.strptime(video_name, '%d-%m-%y_%H-%M-%S')
    files_lisr = []
    for file_name in os.listdir(dir_path):
        timestamp = datetime.strptime(file_name, "%d-%m-%y_%H_M%S")
        files_lisr.append((abs(timestamp-wanted_timestamp), timestamp, file_name))
    delay, timestamp, file_name = sorted(files_lisr)[0]
    print(f"{delay=},{timestamp=},{file_name=}")
    return file_name


def main(args):
    video_name, station, csv_name, video_cap = args.vi, args.s, args.csv, args.vf
    if not station:
        station = "robert"
        if len(video_name.split(".")) > 1:
            station = "sapir"
    if not csv_name:
        csv_name = find_closest(video_name, station)

    make_video_from(video_name=video_name, csv_name=csv_name, station_name=station, computer_date=True,
                    composed_video_suffix="_video_only.mp4", full_video_suffix="_full.mp4", video_cap=video_cap)


if __name__ == '__main__':
    # create_rfid_index()
    parser = argparse.ArgumentParser(
        description="create videos for each wanted tool from conf. merged with the given input")
    parser.add_argument('vi', type=str, help='video input name')
    parser.add_argument('-s', metavar="station name", type=str, help='station name (robert/sapir)')
    parser.add_argument('-csv', metavar="csv path", type=str, help='csv path for rfid scan')
    parser.add_argument('-vf', metavar="video frame", type=str, help='take only vf frame from video',
                        default=float("inf"))
    main(parser.parse_args())
