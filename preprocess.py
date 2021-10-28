import pandas as pd
import os
import datetime
import re
import time
import numpy as np


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
                real_time = int(datetime.datetime.strptime(file_name, "%Y_%m_%d_%H_%M_%S").timestamp())

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
                    values[date_index] = str(datetime.datetime.fromtimestamp(updated_time))
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


NO_SIGNAL_OK_COOLDOWN = 10


def fixing_de_syncs(input_dir_path, output_dir_path):
    def try_int(x):
        try:
            return int(x)
        except ValueError:
            return np.nan

    def try_date(x):
        try:
            datetime.datetime.strptime(file_name, "%Y_%m_%d_%H_%M_%S")
            return True
        except ValueError:
            return False

    def check_desync_groups(df: pd.DataFrame):
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
        print(f"problematic indexes by regex:\n{issues}")

        df["problematic_index"] = (outliers_low | outliers_high | nans | ~good_rows)
        # df.to_csv(path_or_buf=output_file, header=True, index=False)
        df['group'] = df['problematic_index'].ne(df['problematic_index'].shift()).cumsum()
        df["group"] = (df["group"] - 1) // 2
        return df

    antenna_time_multipliers = []
    for file_name in os.listdir(input_dir_path):
        print(f"\n\nworking on {file_name}")
        input_file = os.path.join(input_dir_path, file_name)
        output_file = os.path.join(output_dir_path, file_name)
        df = pd.read_csv(filepath_or_buffer=input_file, header=0, index_col=False, dtype=str)
        df = check_desync_groups(df)

        sync_groups = df.groupby('group')
        time_jumps = df["Time"][df['group'].ne(df['group'].shift(1))]
        start_window_time = int(datetime.datetime.strptime(file_name, "%Y_%m_%d_%H_%M_%S").timestamp())
        for group_index, group in enumerate(sync_groups):
            # end_window_time = time_jumps[]
            print("a")
        print("a")


def main():
    input_dir_path = "lab_unlabeled/axis_bob_21_10"
    output_dir_path = "lab_testing/axis_bob_21_10"
    fixing_de_syncs(input_dir_path, output_dir_path)


if __name__ == '__main__':
    main()
