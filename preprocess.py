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
        df, problematic_indexes = check_desync_groups(df, True)
        sync_groups = df.groupby('group')
        if "computer date" not in df:
            group = pd.DataFrame()
            print(f"{len(sync_groups)=}")
            if len(sync_groups) == 1:
                group = df
            elif len(sync_groups) == 2:
                group = df.iloc[problematic_indexes[0] + 1:, :]
            elif len(sync_groups) > 2:
                temp = np.zeros(shape=(2 + len(problematic_indexes),), dtype=int)
                temp[1:-1] = problematic_indexes.values
                temp[-1] = df.shape[0]
                diff_temp = np.diff(temp)
                index = np.where(diff_temp == diff_temp.max())[0][0]
                group = df.iloc[temp[index]: temp[index + 1], :]

            if group.shape[0] > 50:
                group.to_csv(path_or_buf=os.path.join(output_dir_path, file_name), header=True)
        else:
            start_window_time = datetime.strptime(file_name, "%Y_%m_%d_%H_%M_%S").timestamp()
            group: pd.DataFrame
            good_groups_size = []
            for group_index, group in sync_groups:
                antenna_window_time_diff = int(group.iloc[-1]["Time"]) - int(group.iloc[0]["Time"])
                end_window_time = datetime.strptime(group.iloc[-1]["computer date"].split('.')[0],
                                                    "%Y-%m-%d %H:%M:%S").timestamp()
                real_window_time_diff = end_window_time - start_window_time
                print(f"{antenna_window_time_diff=}, {real_window_time_diff=}")
                if abs(antenna_window_time_diff - real_window_time_diff) <= 1:
                    good_groups_size.append(
                        (group.shape[0], int(group.first_valid_index()), int(group.last_valid_index())))
                antenna_time_multipliers.append((antenna_window_time_diff, real_window_time_diff))
                start_window_time = end_window_time
            if len(good_groups_size) > 0:
                good_groups_size.sort()
                group = df.iloc[good_groups_size[-1][1]: good_groups_size[-1][2] + 1, :]
                group.to_csv(path_or_buf=os.path.join(output_dir_path, file_name), header=True)
    print("Done")


def plot(input_dir_path):
    for file_name in os.listdir(input_dir_path):
        print(f"\n\nworking on {file_name}")
        df = pd.read_csv(filepath_or_buffer=os.path.join(input_dir_path, file_name), dtype=str)
        df = df[["EPC", "Date", "Antenna"]]
        min_freq = max(int(df.shape[0] ** 0.5), 2)
        print(f"df size: {df.shape[0]}, showing EPC with freq greater then {min_freq}")
        epc = df[['EPC']]
        df = df[epc.replace(epc.apply(pd.Series.value_counts)).gt(min_freq).all(1)]
        df = df.astype({'Date': 'datetime64[ns]'})
        colormap = {1: 'red', 2: 'blue'}
        df["Color"] = df["Antenna"].apply(lambda x: colormap.get(int(x), 'red'))
        #TODO: dinf why nones is 8:46:42
        df.plot.scatter(x="Date", y="EPC", title=file_name, alpha=0.3, figsize=(10, 4), color=df.Color)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()


def plot_epc_statistics(input_dir_path):
    def create_merged_df(input_dir_path):
        merged_df = pd.DataFrame()
        for file_name in os.listdir(input_dir_path):
            print(f"\n\nworking on {file_name}")
            round_df = pd.read_csv(filepath_or_buffer=os.path.join(input_dir_path, file_name), index_col=False,
                                   dtype=str)
            round_df = round_df[["EPC", "Time", "Antenna"]]
            round_df["File"] = file_name
            print(f"{round_df.shape=}")
            if round_df.shape[0] < 500:
                print("next!")
                continue
            min_time = int(round_df.iloc[0]["Time"])
            max_time = int(round_df.iloc[-1]["Time"])
            round_df["Time"] = (round_df["Time"].apply(pd.to_numeric) - min_time) / (max_time - min_time)
            merged_df = pd.concat([merged_df, round_df], ignore_index=True)
            print("merged!")
        print("Done merging")
        merged_df.drop_duplicates(inplace=True, ignore_index=True)
        return merged_df

    merged_df = create_merged_df(input_dir_path)

    # creating counter_file df
    c_df = pd.pivot_table(merged_df, values='Antenna', index='File', columns='EPC', fill_value=0, aggfunc=len)

    # creating rounded-time df
    merged_df.loc[:, "r_Time"] = merged_df["Time"].round(3)
    merged_df.sort_values(by=["EPC", "Time"], ignore_index=True, inplace=True)
    round_df = merged_df[merged_df.duplicated(subset=["EPC", "r_Time"], keep=False)]

    # creating windowed df
    w_df = merged_df[["EPC", "Time", "File", "Antenna"]]
    time_test_checker = pd.Series(dtype=bool)
    _w_df = pd.DataFrame(columns=["w_EPC", "w_Time", "w_File", "w_Antenna"])
    for epc_id, df in w_df.groupby("EPC"):
        df[["w_EPC", "w_Time", "w_File"]] = df[["EPC", "Time", "File"]].rolling(3).std()
        df.loc[:, "w_Antenna"] = df["Antenna"].rolling(3).quantile(0.5)
        _w_df = _w_df.append(df[["w_EPC", "w_Time", "w_File", "w_Antenna"]])
        _epc = (df["w_EPC"] <= 0.001) | (df["w_EPC"].isna())
        _time = (df["w_Time"] < 0.01) | (df["w_Time"].isna())
        time_test_checker = time_test_checker.append((_epc & _time))
    w_df.loc[:, "time_tester"] = time_test_checker
    w_df[["w_EPC", "w_Time", "w_File", "w_Antenna"]] = _w_df

    w_df.loc[:, "w_Antenna"] = w_df.apply(
        lambda row: int(row['Antenna']) if np.isnan(row['w_Antenna']) else int(row['w_Antenna']), axis=1)
    # w_df = w_df[w_df["w_File"] > 0.01]
    time_w_df = w_df[w_df["time_tester"]]
    antenna_w_df = w_df[["EPC", "Time", "Antenna", "w_Antenna"]]
    colormap = {1: 'red', 2: 'blue'}
    antenna_w_df.loc[:, "color"] = antenna_w_df["w_Antenna"].apply(lambda x: colormap.get(int(x), 'red'))
    merged_df.loc[:, "color"] = merged_df["Antenna"].apply(lambda x: colormap.get(int(x), 'red'))
    # plotting
    merged_name = input_dir_path.split('\\')[1]
    ax = merged_df.plot(x="Time", y="EPC", kind='scatter', alpha=0.2, marker="o")
    time_w_df.plot(x="Time", y="EPC", kind='scatter', alpha=0.3, marker="s", color='red', ax=ax)
    plt.title(f"{merged_name} Time Filtered")
    plt.show()

    ax = merged_df.plot(x="Time", y="EPC", kind='scatter', alpha=0.2, marker="o", color=merged_df.color)
    antenna_w_df.plot(x="Time", y="EPC", kind='scatter', alpha=0.3, marker="s", color=antenna_w_df.color, ax=ax)
    plt.title(f"{merged_name} antenna Filtered")
    plt.show()


def main():
    dir_path = "new_bob_22_10"
    input_dir_path = os.path.join("lab_unlabeled", dir_path)
    output_dir_path = os.path.join("lab_testing", dir_path)
    save_sync_groups_only(input_dir_path, output_dir_path)
    plot(output_dir_path)

    plot_epc_statistics(output_dir_path)


if __name__ == '__main__':
    main()
