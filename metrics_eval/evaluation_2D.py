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

    print(f'Evaluating patient {patient_name}...')

    results = []
    modes = []

    metrics = sim.ImageMetrics()

    pred_nii_img = nib.load(pred_path)
    pred_nii_array = pred_nii_img.get_fdata()
    original_nii_img = nib.load(original_image_path)
    original_nii_array = original_nii_img.get_fdata()
    mask_nii_img = nib.load(mask_path)
    mask_nii_array = mask_nii_img.get_fdata()

    pred_nii_array = pred_nii_array.transpose()

    mean_mse = 0
    mean_mae = 0
    mean_ssim = 0
    mean_psnr = 0

    for i, slice_pred in enumerate(pred_nii_array):
        slice_pred = np.rot90(slice_pred, 1)
        print(f'Processing slice {i}...')
        print(f'Original shape: {original_nii_array[:, :, i].shape}')
        print(f'Prediction shape: {slice_pred.shape}')
        print(f'Mask shape: {mask_nii_array[:, :, i].shape}')
        metrics_dict = metrics.score_patient(original_nii_array[:, :, i], slice_pred, mask_nii_array[:, :, i])
        if not np.isnan(metrics_dict["mse"]):
            print(f'Patient {patient_name}_{i}: \nMSE: {metrics_dict["mse"]} \nMAE: {metrics_dict["mae"]} \nSSIM:{metrics_dict["ssim"]} \nPSNR: {metrics_dict["psnr"]}')
            results.append([patient_name, i, metrics_dict["mse"], metrics_dict["mae"], metrics_dict["ssim"], metrics_dict["psnr"]])
            mean_mse += metrics_dict["mse"]/len(pred_nii_array)
            mean_mae += metrics_dict["mae"]/len(pred_nii_array)
            mean_ssim += metrics_dict["ssim"]/len(pred_nii_array)
            mean_psnr += metrics_dict["psnr"]/len(pred_nii_array)

    # Append to the results the mean of the patient metrics
    results.append([patient_name, 'Mean', mean_mse, mean_mae, mean_ssim, mean_psnr])
    modes.append([patient_name, 'Mean', mean_mse, mean_mae, mean_ssim, mean_psnr])
    
    return results, modes


def evaluate_all_patients(preds_path, original_images_paths, directory, exp_name='2D_eval'):
    '''
    This function evaluates the metrics of all the patients in the predictions directory.
    '''

    # Get the list of the predictions list
    preds_list = os.listdir(preds_path)

    patient_metrics = [['Patient', 'Slice', 'MSE', 'MAE', 'SSIM', 'PSNR']]
    modes = []

    for pred in preds_list:
        patient_name = pred.split('_')[0]
        pred_path = os.path.join(preds_path, pred)
        og_image_path = os.path.join(original_images_paths, patient_name, 'ct.nii.gz')
        mask_path = os.path.join(original_images_paths, patient_name, 'mask.nii.gz')
        res, mod = evaluate_patient(pred_path, og_image_path, mask_path, patient_name)
        for r in res:
            patient_metrics.append(r)
        for m in mod:
            modes.append(m)

    # Compute the min, 1quantile, median, 3quantile and max of the modes. Computer the mean of the modes. Append to the end of patient_metrics
    mse_modes_min = np.min([m[2] for m in modes])
    mae_modes_min = np.min([m[3] for m in modes])
    ssim_modes_min = np.min([m[4] for m in modes])
    psnr_modes_min = np.min([m[5] for m in modes])

    mse_modes_1q = np.percentile([m[2] for m in modes], 25)
    mae_modes_1q = np.percentile([m[3] for m in modes], 25)
    ssim_modes_1q = np.percentile([m[4] for m in modes], 25)
    psnr_modes_1q = np.percentile([m[5] for m in modes], 25)

    mse_modes_med = np.median([m[2] for m in modes])
    mae_modes_med = np.median([m[3] for m in modes])
    ssim_modes_med = np.median([m[4] for m in modes])
    psnr_modes_med = np.median([m[5] for m in modes])

    mse_modes_3q = np.percentile([m[2] for m in modes], 75)
    mae_modes_3q = np.percentile([m[3] for m in modes], 75)
    ssim_modes_3q = np.percentile([m[4] for m in modes], 75)
    psnr_modes_3q = np.percentile([m[5] for m in modes], 75)

    mse_modes_max = np.max([m[2] for m in modes])
    mae_modes_max = np.max([m[3] for m in modes])
    ssim_modes_max = np.max([m[4] for m in modes])
    psnr_modes_max = np.max([m[5] for m in modes])

    mse_modes_mean = np.mean([m[2] for m in modes])
    mae_modes_mean = np.mean([m[3] for m in modes])
    ssim_modes_mean = np.mean([m[4] for m in modes])
    psnr_modes_mean = np.mean([m[5] for m in modes])

    patient_metrics.append(['Modes', 'Min', mse_modes_min, mae_modes_min, ssim_modes_min, psnr_modes_min])
    patient_metrics.append(['Modes', '1Q', mse_modes_1q, mae_modes_1q, ssim_modes_1q, psnr_modes_1q])
    patient_metrics.append(['Modes', 'Med', mse_modes_med, mae_modes_med, ssim_modes_med, psnr_modes_med])
    patient_metrics.append(['Modes', '3Q', mse_modes_3q, mae_modes_3q, ssim_modes_3q, psnr_modes_3q])
    patient_metrics.append(['Modes', 'Max', mse_modes_max, mae_modes_max, ssim_modes_max, psnr_modes_max])
    patient_metrics.append(['Modes', 'Mean', mse_modes_mean, mae_modes_mean, ssim_modes_mean, psnr_modes_mean])

    # Save the metDrics in a c.s.v file and inside the directory folder.
    with open(f'{directory}/{exp_name}_patient_metrics.csv', 'w') as f:
        for patient in patient_metrics:
            f.write(f'{patient[0]},{patient[1]},{patient[2]},{patient[3]},{patient[4]},{patient[5]}\n')

    return patient_metrics