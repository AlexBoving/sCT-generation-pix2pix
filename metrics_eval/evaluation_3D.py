'''
This script uses the nii.gz files from the predictions, the original files cbct.nii.gz and the masks.nii.gz files
to evaluate the metrics of the patients.
'''

import os
import numpy as np
import nibabel as nib
import image_metrics as sim
import matplotlib.pyplot as plt

def evaluate_patient(pred_path, original_image_path, mask_path, patient_name):
    '''
    This function evaluates the metrics of a patient.
    '''

    # This is the path of the patient: "/Users/alexboving/Desktop/Thèse/synthrad-pix2pix/Testing/Task1_8bit_stacked/brain_training_test_8bit_stacked_eval/1BB100_ct_prediction.nii.gz"
    # I want to print the patient name: 1BB100
    print(f'Evaluating patient {patient_name}...')

    metrics = sim.ImageMetrics()

    pred_nii_img = nib.load(pred_path)
    original_nii_img = nib.load(original_image_path)
    mask_nii_img = nib.load(mask_path)

    pred_nii_array = pred_nii_img.get_fdata()
    original_nii_array = original_nii_img.get_fdata()
    mask_nii_array = mask_nii_img.get_fdata()

    # rotate the images -90 degrees
    pred_nii_array = np.rot90(pred_nii_array, 3)
    original_nii_array = np.rot90(original_nii_array, 3)
    mask_nii_array = np.rot90(mask_nii_array, 3)

    """
    # Clip the values of the prediction and the original image to [-1000, 1000]
    pred_nii_array = np.clip(pred_nii_array, -1000, 1000)
    original_nii_array = np.clip(original_nii_array, -1000, 1000)

    # Apply the mask to the original image. All the values outside the mask will be -1000.
    original_nii_array = np.where(mask_nii_array == 0, -1000, original_nii_array)
    """

    """
    # Plot the images of idx 90:
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs[0].imshow(original_nii_array[:, :, 90], cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(pred_nii_array[:, :, 90], cmap='gray')
    axs[1].set_title('Predicted Image')
    plt.show()

    # Plot figure of the prediction
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.imshow(pred_nii_array[:, :, 90], cmap='gray')
    axs.set_title('Predicted Image')
    plt.show()
    """

    metrics_dict = metrics.score_patient(original_nii_array, pred_nii_array, mask_nii_array)
    print(f'Patient {patient_name}: \nMSE: {metrics_dict["mse"]} \nMAE: {metrics_dict["mae"]} \nSSIM:{metrics_dict["ssim"]} \nPSNR: {metrics_dict["psnr"]}')

    return metrics_dict["mse"], metrics_dict["mae"], metrics_dict["ssim"], metrics_dict["psnr"]


def evaluate_all_patients(preds_path, original_images_paths, directory, exp_name='3D_eval'):
    '''
    This function evaluates the metrics of all the patients in the predictions directory.
    '''

    # Get the list of the predictions list
    preds_list = os.listdir(preds_path)

    patient_metrics = [['Patient', 'MSE', 'MAE', 'SSIM', 'PSNR']]

    for pred in preds_list:
        patient_name = pred.split('_')[0]
        pred_path = os.path.join(preds_path, pred)
        og_image_path = os.path.join(original_images_paths, patient_name, 'ct.nii.gz')
        mask_path = os.path.join(original_images_paths, patient_name, 'mask.nii.gz')
        mse, mae, ssim, psnr = evaluate_patient(pred_path, og_image_path, mask_path, patient_name)
        patient_metrics.append([patient_name, mse, mae, ssim, psnr])

    # Append to the results the mean of the patients metrics
    mse_min = np.min([patient[1] for patient in patient_metrics[1:]])
    mae_min = np.min([patient[2] for patient in patient_metrics[1:]])
    ssim_min = np.min([patient[3] for patient in patient_metrics[1:]])
    psnr_min = np.min([patient[4] for patient in patient_metrics[1:]])

    mse_1quartile = np.percentile([patient[1] for patient in patient_metrics[1:]], 25)
    mae_1quartile = np.percentile([patient[2] for patient in patient_metrics[1:]], 25)
    ssim_1quartile = np.percentile([patient[3] for patient in patient_metrics[1:]], 25)
    psnr_1quartile = np.percentile([patient[4] for patient in patient_metrics[1:]], 25)

    mse_median = np.median([patient[1] for patient in patient_metrics[1:]])
    mae_median = np.median([patient[2] for patient in patient_metrics[1:]])
    ssim_median = np.median([patient[3] for patient in patient_metrics[1:]])
    psnr_median = np.median([patient[4] for patient in patient_metrics[1:]])

    mse_3quartile = np.percentile([patient[1] for patient in patient_metrics[1:]], 75)
    mae_3quartile = np.percentile([patient[2] for patient in patient_metrics[1:]], 75)
    ssim_3quartile = np.percentile([patient[3] for patient in patient_metrics[1:]], 75)
    psnr_3quartile = np.percentile([patient[4] for patient in patient_metrics[1:]], 75)

    mse_max = np.max([patient[1] for patient in patient_metrics[1:]])
    mae_max = np.max([patient[2] for patient in patient_metrics[1:]])
    ssim_max = np.max([patient[3] for patient in patient_metrics[1:]])
    psnr_max = np.max([patient[4] for patient in patient_metrics[1:]])

    mse_mean = np.mean([patient[1] for patient in patient_metrics[1:]])
    mae_mean = np.mean([patient[2] for patient in patient_metrics[1:]])
    ssim_mean = np.mean([patient[3] for patient in patient_metrics[1:]])
    psnr_mean = np.mean([patient[4] for patient in patient_metrics[1:]])

    patient_metrics.append(['Min', mse_min, mae_min, ssim_min, psnr_min])
    patient_metrics.append(['1st Quartile', mse_1quartile, mae_1quartile, ssim_1quartile, psnr_1quartile])
    patient_metrics.append(['Median', mse_median, mae_median, ssim_median, psnr_median])
    patient_metrics.append(['3rd Quartile', mse_3quartile, mae_3quartile, ssim_3quartile, psnr_3quartile])
    patient_metrics.append(['Max', mse_max, mae_max, ssim_max, psnr_max])
    patient_metrics.append(['Mean', mse_mean, mae_mean, ssim_mean, psnr_mean])

    # Save the metDrics in a c.s.v file
    with open(f'{directory}/{exp_name}_patient_metrics.csv', 'w') as f:
        for patient in patient_metrics:
            f.write(f'{patient[0]},{patient[1]},{patient[2]},{patient[3]},{patient[4]}\n')

    return patient_metrics