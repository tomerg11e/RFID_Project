import pandas as pd
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from typing import Iterable
import seaborn as sns

SYRINGES_GROUP = 'syringes'
BREATHING_GROUP = 'breathing'
MASK_GROUP = 'mask'
MEDICATION_GROUP = 'medication'
AIRWAY_GROUP = 'airway'
BOARD_GROUP = 'board'
TOOL_GROUPS_NUM = 6


def check_de_sync_groups(df: pd.DataFrame, print_bool=True, no_signal_ok_cooldown=10):
    """
    check for valid rows in the df by delta time and regex
    :param df:
    :param print_bool:
    :param no_signal_ok_cooldown:
    :return:    the df with new column sync_group for helping distinct between buffer and data in the csv file
                and the problematic indexes
    """

    def try_int(x):
        try:
            return int(x)
        except ValueError:
            return np.nan

    def try_date(x):
        try:
            datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            return True
        except ValueError:
            return False

    # <editor-fold desc="Delta time">
    df["delta_Time"] = df["Time"].apply(try_int) - df["Time"].shift(-1).apply(try_int)
    q1 = df["delta_Time"].quantile(0.25)
    q3 = df["delta_Time"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = min(-no_signal_ok_cooldown, q1 - 1.5 * iqr)
    upper_bound = min(0, q3 + 1.5 * iqr)
    outliers_low = (df["delta_Time"] < lower_bound)
    outliers_high = (df["delta_Time"] > upper_bound)
    nans = (df["delta_Time"][:-1].isna())
    delta_time_issues = df["delta_Time"][outliers_low | outliers_high | nans]
    # </editor-fold>
    if print_bool:
        print(f"problematic rows by delta time with bounds of: {lower_bound=}, {upper_bound=}")
        print(delta_time_issues)
        print(f"size of original df {df.shape}")

    # <editor-fold desc="regex checks">
    good_epc = df["EPC"].str.contains(r'^(?:3\d){4}$')
    good_time = df["Time"].str.contains(r'^\d{10}$')
    good_read_count = df["ReadCount"].str.contains(r'^\d$')
    good_rssi = df["RSSI"].apply(try_int).between(-100, 0)
    good_antenna = df["Antenna"].str.contains(r'^(?:1|2)$')
    good_frequency = (df["Frequency"].str.contains(r'^\d{6}$'))
    good_phase = df["Phase"].apply(try_int).between(0, 180)
    good_date = (df["Date"].apply(try_date))
    good_rows = good_epc & good_time & good_read_count & good_rssi & good_antenna \
                & good_frequency & good_phase & good_date
    issues = df[~good_rows]
    # </editor-fold>
    if print_bool:
        print(f"problematic indexes by regex:\n{issues}")

    df["problematic_index"] = (outliers_low | outliers_high | nans | ~good_rows)
    # df.to_csv(path_or_buf=output_file, header=True, index=False)
    problematic_indexes = df[df["problematic_index"]].index
    df["sync_group"] = df['problematic_index'].ne(df['problematic_index'].shift()).cumsum()
    df["sync_group"] = (df["sync_group"] - 1) // 2
    df.drop('problematic_index', axis=1, inplace=True)
    return df, problematic_indexes


def plot_rssi_df(df: pd.DataFrame, title: str = 'RSSI_std', show: bool = True, ax: plt.Axes = None, antenna=1):
    df = df[df["Antenna"] == antenna].copy()
    df.loc[:, "w_RSSI"] = df["RSSI"].rolling(5, center=True).std()
    df["order"] = df["EPC"].apply(lambda x: int(x))
    df.sort_values(by=['order'], inplace=True)
    y_column = 'EPC'
    if "Tool_name" in df.columns:
        y_column = 'Tool_name'
    ax = sns.scatterplot(x='Date', y=y_column, hue='w_RSSI',
                         data=df, legend=False, ax=ax, s=40)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.set_xlabel("Time")
    ax.set_title(title)
    if show:
        plt.show()
    else:
        return ax


def plot_rssi_file(file_path, show: bool = True, ax: plt.Axes = None, antenna: int = 1):
    print(f"\nworking on {file_path}")
    df = pd.read_csv(filepath_or_buffer=file_path, dtype=str)[["EPC", "Time", "Date", "Antenna", "RSSI"]]
    df = df.astype({'Date': 'datetime64', 'RSSI': 'int', 'Antenna': 'int', 'Time': 'int'})
    file_path = file_path.split('\\')[1]
    title = f"RSSI_std_{file_path}"
    return plot_rssi_df(df, title, show, ax, antenna)


def filter_df_by_freq(df: pd.DataFrame) -> pd.DataFrame:
    """
    filter the given df by returning only the most recurrent epcs from each group
    :param df:
    :return:
    """
    group_df = pd.read_csv(filepath_or_buffer="helper files/rfid_index.csv", dtype=str, index_col="EPC")

    vc = df["EPC"].value_counts().to_frame().reset_index().astype('int').rename(columns={"EPC": "Freq", "index": "EPC"})
    vc = vc.join(group_df, on="EPC")
    vc = vc[vc["Group"] != "_"]
    # vc = vc[vc.groupby(['Group'])['Freq'].transform(max) == df['Freq']]
    g_vc = vc.groupby('Group').agg({"Freq": 'max'}).reset_index(col_fill='Group')
    vc = pd.merge(vc, g_vc, on=['Freq', 'Group'])
    epcs = list(vc["EPC"].astype('str'))
    breathing_vc = vc[vc["Group"] == BREATHING_GROUP]["EPC"]
    if len(breathing_vc) > 0:
        breathing_epc = vc[vc["Group"] == BREATHING_GROUP]["EPC"].item()
        other_breathing_epc = breathing_epc - 1 if breathing_epc % 2 else breathing_epc + 1
        epcs.append(str(other_breathing_epc))
    df = df[df["EPC"].isin(epcs)].astype({"EPC": 'int'})
    df = df.join(group_df, on="EPC")
    return df


def plot_line_rssi_df(df: pd.DataFrame, title: str = "RSSI_lines", show: bool = False, axes: Iterable[plt.Axes] = None,
                      antenna: int = 1):
    """
    creating RSSI-plot for each wanted group
    :param df:
    :param title:
    :param show:
    :param axes:
    :param antenna:
    :return:
    """
    df = df[df["Antenna"] == antenna].copy()
    num_groups = len(df["Group"].unique())
    if axes is None:
        fig, axes = plt.subplots(figsize=(19, 5), ncols=1, nrows=num_groups)

    y_column = 'EPC'
    if "Tool_name" in df.columns:
        y_column = 'Tool_name'
    df["sync_group"] = df["sync_group"].astype('str')
    rssi_min = df["RSSI"].min()
    rssi_max = df["RSSI"].max()
    for i, (group_name, grouped_df) in enumerate(df.groupby("Group")):
        max_group = grouped_df["sync_group"].value_counts().index[0]
        markers = {'max': "o", 'other': "X"}
        grouped_df["marker_style"] = grouped_df["sync_group"].apply(lambda x: 'max' if x == max_group else 'other')
        ax = sns.lineplot(x='Date', y='RSSI', data=grouped_df, hue=y_column, markers=markers, ax=axes[i],
                          legend=False, style='marker_style')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.set_xlabel("Time")
        y_label = str(grouped_df["Tool_name"].iloc[0]) + ", " + str(grouped_df["EPC"].iloc[0])
        ax.set_ylabel(y_label)
        ax.set_title(f"{title}_{group_name}")
        ax.set_ylim([rssi_min, rssi_max])
    if show:
        plt.show()
    else:
        return axes


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
