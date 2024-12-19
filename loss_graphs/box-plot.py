import numpy as np
import matplotlib.pyplot as plt

def box_plot_mae(data, labels):
    """
    Create a box plot of MAE values for multiple experiments.

    :param data: A list of lists containing MAE values for each experiment.
    :param labels: A list of strings containing the labels for each experiment.
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, patch_artist=True, labels=labels, whis=[0, 100])
    plt.title('Mean Absolute Error (MAE) Box Plot')
    plt.ylabel('MAE Value')
    plt.grid(True)
    plt.show()

def box_plot_mse(data, labels):
    """
    Create a box plot of MSE values for multiple experiments.

    :param data: A list of lists containing MSE values for each experiment.
    :param labels: A list of strings containing the labels for each experiment.
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, patch_artist=True, labels=labels, whis=[0, 100])
    plt.ylabel('MSE Value')
    plt.grid(True)
    plt.show()

def box_plot_psnr(data, labels):
    """
    Create a box plot of PSNR values for multiple experiments.

    :param data: A list of lists containing PSNR values for each experiment.
    :param labels: A list of strings containing the labels for each experiment.
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, patch_artist=True, labels=labels, whis=[0, 100])
    plt.ylabel('PSNR Value')
    plt.grid(True)
    plt.show()

def box_plot_ssim(data, labels):
    """
    Create a box plot of SSIM values for multiple experiments.

    :param data: A list of lists containing SSIM values for each experiment.
    :param labels: A list of strings containing the labels for each experiment.
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, patch_artist=True, labels=labels, whis=[0, 100])
    plt.ylabel('SSIM Value')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    labels=['8bit grayscale', 'RGB w/o placem.', 'RGB w/ placem.', '16bit grayscale', '3 stacked slices', '5 stacked slices']
    # ---------
    mse_data = [
        [35373.89, 62740.85, 71795.35, 84003.72, 102183.95], # 8bit
        [92875.12, 99269.08, 140097.86, 169060.41, 222328.58], # rgb - without bit placement.
        [70243.37, 83283.11, 87432.05, 91276.65, 103196.81], # rgb - with bit placement.
        [35830.43, 63028.06, 68158.94, 87428.88, 124772.83], # 16bit
        [29236.45, 55247.26, 67953.83, 88112.74, 130442.22], # 3 stacked 16bit slices.
        [27799.61, 57793.16, 66221.82, 82011.37, 121575.25]  # 5 stacked 16bit slices.
    ]

    mae_data = [
        [102.37, 134.23, 150.72, 165.11, 180.34], # 8bit
        [115.32, 125.55, 146.06, 175.28, 189.20], # rgb - without bit placement
        [160.19, 188.07, 195.11, 198.71, 210.15], # rgb - with bit placement
        [96.14, 122.50, 143.64, 165.65, 190.58], # 16bit
        [86.68, 115.10, 143.18, 163.91, 191.45], # 3 stacked 16bit slices.
        [83.82, 114.70, 139.23, 157.06, 185.66]  # 5 stacked 16bit slices.
    ]

    psnr_data = [
        [21.99, 22.85, 23.53, 24.11, 26.60], # 8bit.
        [18.62, 19.81, 20.63, 22.13, 22.41], # rgb - without bit placement.
        [21.95, 22.48, 22.67, 22.88, 23.62], # rgb - with bit placement.
        [21.31, 22.67, 23.75, 24.0, 26.55], # 16bit.
        [20.93, 22.64, 23.77, 24.67, 27.43], # 3 stacked 16bit slices.
        [21.92, 23.70, 24.67, 25.28, 28.59]  # 5 stacked 16bit slices.
    ]

    ssim_data = [
        [0.675, 0.706, 0.727, 0.735, 0.837], # 8bit.
        [0.6185, 0.6444, 0.6820, 0.7115, 0.7389], # rgb - without bit placement.
        [0.5459, 0.5706, 0.581, 0.5992, 0.6433], # rgb - with bit placement.
        [0.672, 0.710, 0.740, 0.752, 0.842],  # 16bit.
        [0.686, 0.711, 0.748, 0.759, 0.861], # 3 stacked 16bit slices.
        [0.716, 0.735, 0.772, 0.792, 0.894]  # 5 stacked 16bit slices
    ]

    box_plot_mse(mse_data, labels)
    box_plot_mae(mae_data, labels)
    box_plot_psnr(psnr_data, labels)
    box_plot_ssim(ssim_data, labels)
