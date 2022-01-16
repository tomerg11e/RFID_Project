import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2

plt.switch_backend('Agg')

CALIBRATION_MATRIX = np.array(
    [[-0.015031657182, -0.006924165878, -0.033659838140, 2.192849397659, -0.058368790895, -2.201693058014],
     [-0.026684835553, -2.594270467758, 0.003967078403, 1.296077728271, 0.013595988043, 1.275859475136],
     [1.475889682770, -0.006595089100, 1.498858213425, -0.023269193247, 1.520210146904, -0.019803291187],
     [-0.114770732820, -11.348546028137, 21.927616119385, 5.393544197083, -21.963207244873, 5.771105289459],
     [-24.714830398560, -0.013862852007, 13.093193054199, -9.735026359558, 12.893286705017, 9.520476341248],
     [-0.106131777167, -18.179416656494, 0.242432817817, -18.064281463623, -0.603124320507, -18.489496231079]])


def calibrate_ai(ai):
    '''
    load-cell calibration

    param ai: numpy array
        matrix of 16 load cell measurments "dev1/ai#"

    return: numpy array with 16 rows matrix with 6 rows, fx fy fz tx ty tz
    '''
    sg = ai[:, :6] - ai[:, 8:14]
    bias = sg[0, :]
    bs = sg - bias
    tau = np.matmul(CALIBRATION_MATRIX, bs.T)
    return tau


def get_force_torque(df_load_cell):
    # load cell calibration
    load_cell_data = df_load_cell.iloc[:, 3:].to_numpy()
    tau = calibrate_ai(load_cell_data)

    total_force = np.sqrt(tau[0, :] ** 2 + tau[1, :] ** 2 + tau[2, :] ** 2)
    total_torque = np.sqrt(tau[3, :] ** 2 + tau[4, :] ** 2 + tau[5, :] ** 2)

    return total_force, total_torque


def video_writer(input_vid: str, output_vid: str, df_load_cell, df_straingage):

    left_hand_col = df_straingage.columns[2]
    right_hand_col = df_straingage.columns[3]

    # plot figure
    timestamp = 0
    fig = plt.figure(figsize =(15, 7))
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4]) 

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Force/Torque Load Cell')
    ax1.legend()
    
    ax2.set_xlabel('Time (m/s)')
    ax2.set_ylabel('Force Strain Gage')
    
    ax1.plot(df_load_cell.absolute_time, total_torque, color='tab:orange', linewidth = 1, label='Torque')
    ax1.plot(df_load_cell.absolute_time, total_force, color='tab:blue', linewidth = 1, label='Force')
    
    ax2.plot(df_straingage.absolute_time, df_straingage[left_hand_col], color='tab:blue', linewidth = 1)
    ax2.plot(df_straingage.absolute_time, df_straingage[right_hand_col], color='tab:orange', linewidth = 1)

    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # animated=True tells matplotlib to only draw the artist when we explicitly request it
    line1 = ax1.axvline(x=timestamp, color = 'black', animated=True)
    line2 = ax2.axvline(x=timestamp, color = 'black', animated=True)
        
    # cache the axes backgrounds
    bg_cache1 = fig.canvas.copy_from_bbox(ax1.bbox)
    bg_cache2 = fig.canvas.copy_from_bbox(ax2.bbox)
    
    fig.canvas.restore_region(bg_cache1)
    fig.canvas.restore_region(bg_cache2)
    
    fig.canvas.draw()
    
    cap = cv2.VideoCapture(input_vid)
    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_vid, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (1080, 928))

    while (cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            
            # cache the axes background for the next round
            bg_cache1 = fig.canvas.copy_from_bbox(ax1.bbox)
            bg_cache2 = fig.canvas.copy_from_bbox(ax2.bbox)
            
            # update the artist
            line1.set_xdata(timestamp)
            line2.set_xdata(line1.get_xdata())
            
            # re-render the artist
            ax1.draw_artist(line1)
            ax2.draw_artist(line2)
            
            fig.canvas.blit(ax1.bbox)
            fig.canvas.blit(ax2.bbox)
            
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image_from_plot = image_from_plot[50:-30,:,:]
            
            pad_width = (image_from_plot.shape[1]-curr_frame.shape[1])//2
            curr_frame_padded = np.pad(curr_frame, pad_width=((12,12),(pad_width,pad_width),(0,0)) ,constant_values=255)
            
            new_frame = cv2.vconcat([curr_frame_padded, image_from_plot])
            out.write(cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR))
            
            fig.canvas.flush_events()
            
            fig.canvas.restore_region(bg_cache1)
            fig.canvas.restore_region(bg_cache2)
        else:
            break

    cap.release()
    out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-L', type=str, help='load cell csv destination', required=True)
    parser.add_argument('-S', type=str, help='strain gage csv destination', required=True)
    parser.add_argument('-V', type=str, help='top video file destination', required=True)
    args = parser.parse_args()

    # load cell
    load_cell_file = args.L
    df_load_cell = pd.read_csv(load_cell_file, skiprows=21)  # first 21 rows are garbage
    df_load_cell['absolute_time'] = df_load_cell['SAMPLE INDEX'] / 1000
    total_force, total_torque = get_force_torque(df_load_cell)

    # strain gage
    strain_gage_file = args.S
    df_strain_gage = pd.read_csv(strain_gage_file)
    time_bias = df_strain_gage['Scan #'][0] / 100
    df_strain_gage['absolute_time'] = df_strain_gage['Scan #'] / 100 - time_bias  # absolute time in ms

    video_writer(args.V, 'output.avi', df_load_cell, df_strain_gage)
