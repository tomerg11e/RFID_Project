import pandas as pd
import os
from datetime import datetime
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def fixing_jump_de_syncs(input_dir_path, output_dir_path):
    def next_values(input_file, time_delta, de_syncs):
        line = input_file.readline()[:-1]
        values = line.split(',')
        if line == '':
            return '', '', '', ''
        if "EPC" in line or len(re.findall(r'3\d3\d3\d3\d', line)) > 1 or len(values) > len(header_dict):
            de_syncs += 1
            print(f"\tencounter de-sync, total de-sync in this file: {de_syncs}")
            output_file.write('***encountered de-sync***\n')
            if de_syncs > 1:
                print(f"\t***\n\tmore then one de-sync in file {file_name}\n\t***")
            time_delta = 0
            line, values, time_delta, de_syncs = next_values(input_file, time_delta, de_syncs)
        return line, values, time_delta, de_syncs

    for file_name in os.listdir(input_dir_path):
        print(f"\nworking on {file_name}")
        with open(os.path.join(input_dir_path, file_name), 'r') as input_file:
            with open(os.path.join(output_dir_path, file_name), 'w') as output_file:
                header = input_file.readline()[:-1]
                output_file.write(header + "\n")
                header_dict = dict((label, i) for i, label in enumerate(header.split(',')))
                time_index = int(header_dict["Time"])
                date_index = int(header_dict["Date"])
                real_time = int(datetime.strptime(file_name, "%Y_%m_%d_%H_%M_%S").timestamp())

                line = input_file.readline()[:-1]
                values = line.split(',')
                antenna_time = int(values[time_index])
                time_delta = real_time - antenna_time
                print(f"\t{time_delta=}")
                de_syncs = 0
                last_timestamp = real_time
                while True:
                    updated_time = int(values[time_index]) + time_delta
                    values[time_index] = str(updated_time)
                    values[date_index] = str(datetime.fromtimestamp(updated_time))
                    output_file.write(','.join(values) + '\n')

                    line = input_file.readline()[:-1]
                    values = line.split(',')
                    if line == '':
                        break
                    if "EPC" in line or len(re.findall(r'3\d3\d3\d3\d', line)) > 1 or len(values) > len(header_dict):
                        de_syncs += 1
                        print(f"\tencounter de-sync, total de-sync in this file: {de_syncs}")
                        output_file.write('***encountered de-sync***\n')
                        if de_syncs > 1:
                            print(f"\t***\n\tmore then one de-sync in file {file_name}\n\t***")
                        time_delta = 0
                        line = input_file.readline()[:-1]
                        values = line.split(',')


def save_sync_groups_only(input_dir_path, output_dir_path):
    NO_SIGNAL_OK_COOLDOWN = 10

    def try_int(x):
        try:
            return int(x)
        except ValueError:
            return np.nan

    def try_date(x):
        try:
            datetime.strptime(file_name, "%Y_%m_%d_%H_%M_%S")
            return True
        except ValueError:
            return False

    def check_desync_groups(df: pd.DataFrame, print_bool=True):
        df["delta_Time"] = df["Time"].apply(try_int) - df["Time"].shift(-1).apply(try_int)
        q1 = df["delta_Time"].quantile(0.25)
        q3 = df["delta_Time"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = min(-NO_SIGNAL_OK_COOLDOWN, q1 - 1.5 * iqr)
        upper_bound = min(0, q3 + 1.5 * iqr)
        outliers_low = (df["delta_Time"] < lower_bound)
        outliers_high = (df["delta_Time"] > upper_bound)
        nans = (df["delta_Time"][:-1].isna())
        delta_time_issues = df["delta_Time"][outliers_low | outliers_high | nans]
        if print_bool:
            print(f"problematic rows by delta time with bounds of: {lower_bound=}, {upper_bound=}")
            print(delta_time_issues)
            print(f"size of original df {df.shape}")

        good_epc = df["EPC"].str.contains(r'^(?:3\d){4}$')
        good_time = df["Time"].str.contains(r'^\d{10}$')
        # TODO: understand why are we getting more then one readCount. need to be as follows:
        # good_read_count = df["ReadCount"].str.contains(r'^1$')
        good_read_count = df["ReadCount"].str.contains(r'^\d$')
        good_rssi = df["RSSI"].apply(try_int).between(-100, 0)
        good_antenna = df["Antenna"].str.contains(r'^(?:1|2)$')
        good_frequency = (df["Frequency"].str.contains(r'^\d{6}$'))
        good_phase = df["Phase"].apply(try_int).between(0, 180)
        good_date = (df["Date"].apply(try_date))
        good_rows = good_epc & good_time & good_read_count & good_rssi \
                    & good_antenna & good_frequency & good_phase & good_date
        issues = df[~good_rows]
        if print_bool:
            print(f"problematic indexes by regex:\n{issues}")

        df["problematic_index"] = (outliers_low | outliers_high | nans | ~good_rows)
        # df.to_csv(path_or_buf=output_file, header=True, index=False)
        problematic_indexes = df[df["problematic_index"]].index
        df['group'] = df['problematic_index'].ne(df['problematic_index'].shift()).cumsum()
        df["group"] = (df["group"] - 1) // 2
        df.drop('problematic_index', axis=1, inplace=True)
        return df, problematic_indexes

    antenna_time_multipliers = []  # contains tuples of antenna time diff and real time diff
    for file_name in os.listdir(input_dir_path):

        print(f"\n\nworking on {file_name}")
        df = pd.read_csv(filepath_or_buffer=os.path.join(input_dir_path, file_name), index_col=False, dtype=str)
        if "Computer Date" in df:
            df.rename(columns={"Computer Date": "computer date"}, inplace=True)
        if "computer date" not in df:
            continue
        df, problematic_indexes = check_desync_groups(df, True)
        sync_groups = df.groupby('group')
        # time_jumps = df["Time"][df['group'].ne(df['group'].shift(1))]
        start_window_time = datetime.strptime(file_name, "%Y_%m_%d_%H_%M_%S").timestamp()
        group: pd.DataFrame
        saved_csv = False
        for group_index, group in sync_groups:
            antenna_window_time_diff = int(group.iloc[-1]["Time"]) - int(group.iloc[0]["Time"])
            end_window_time = datetime.strptime(group.iloc[-1]["computer date"].split('.')[0],
                                                "%Y-%m-%d %H:%M:%S").timestamp()
            real_window_time_diff = end_window_time - start_window_time
            print(f"{antenna_window_time_diff=}, {real_window_time_diff=}")
            if abs(antenna_window_time_diff - real_window_time_diff) <= 1 and group.shape[0] > 50 and not saved_csv:
                group.to_csv(path_or_buf=os.path.join(output_dir_path, file_name), header=True)
                saved_csv = True
            antenna_time_multipliers.append((antenna_window_time_diff, real_window_time_diff))
            start_window_time = end_window_time

    print("Done")


def plot(input_dir_path):
    for file_name in os.listdir(input_dir_path):
        print(f"\n\nworking on {file_name}")
        df = pd.read_csv(filepath_or_buffer=os.path.join(input_dir_path, file_name), index_col=False, dtype=str)
        df = df[["EPC", "Date"]]
        min_freq = max(int(df.shape[0] ** 0.5), 2)
        print(f"df size: {df.shape[0]}, showing EPC with freq greater then {min_freq}")
        epc = df[['EPC']]
        df = df[epc.replace(epc.apply(pd.Series.value_counts)).gt(min_freq).all(1)]
        df = df.astype({'Date': 'datetime64[ns]'})

        df.plot.scatter(x="Date", y="EPC", title=file_name, alpha=0.3, figsize=(10, 4))
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()


def plot_epc_statistics(input_dir_path):
    merged_df = pd.DataFrame()
    for file_name in os.listdir(input_dir_path):
        print(f"\n\nworking on {file_name}")
        df = pd.read_csv(filepath_or_buffer=os.path.join(input_dir_path, file_name), index_col=False, dtype=str)
        df = df[["EPC", "Time"]]
        print(f"{df.shape=}")
        if df.shape[0] < 500:
            print("next!")
            continue
        min_time = int(df.iloc[0]["Time"])
        max_time = int(df.iloc[-1]["Time"])
        df["Time"] = (df["Time"].apply(pd.to_numeric) - min_time) / (max_time - min_time)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        print("merged!")
    print("Done")
    # plt.rc('xtick', labelsize=24)
    # plt.rc('ytick', labelsize=18)
    merged_df["Time"] = merged_df["Time"].round() #TODO
    df = merged_df.sort_values(by=["EPC", "Time"])
    merged_df.plot(x="Time", y="EPC", kind='scatter', title=input_dir_path, alpha=0.3, marker="o")
    plt.tight_layout()
    plt.show()


def main():
    dir_path = "axis_bob_21_10"
    input_dir_path = os.path.join("lab_unlabeled", dir_path)
    output_dir_path = os.path.join("lab_testing", dir_path)
    # save_sync_groups_only(input_dir_path, output_dir_path)
    # plot(output_dir_path)

    plot_epc_statistics(output_dir_path)


if __name__ == '__main__':
    main()
