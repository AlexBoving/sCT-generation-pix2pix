'''
This script uses the nii.gz files from the predictions, the original files cbct.nii.gz and the masks.nii.gz files
to evaluate the metrics of the patients.
'''

import os
import numpy as np
import nibabel as nib
from PIL import Image
import synthRad2023_image_metrics as sim


def evaluate_patient(pred_path, original_image_path, mask_path):
    '''
    This function evaluates the metrics of a patient.
    '''

    metrics = sim.ImageMetrics()
    metrics_dict = metrics.score_patient(original_image_path, pred_path, mask_path)
    print(f'Patient {pred_path}: \nMAE: {metrics_dict["mae"]} \nSSIM:{metrics_dict["ssim"]} \nPSNR: {metrics_dict["psnr"]}')
    return metrics_dict["mae"], metrics_dict["ssim"], metrics_dict["psnr"]


def evaluate_all_patients(preds_path, original_images_paths, exp_name='exp'):
    '''
    This function evaluates the metrics of all the patients in the predictions directory.
    '''

    # Get the list of the predictions list
    preds_list = os.listdir(preds_path)

    patient_metrics = []

    for pred in preds_list:
        patient_name = pred.split('_')[0]
        pred_path = os.path.join(preds_path, pred)
        og_image_path = os.path.join(original_images_paths, patient_name, 'ct.nii.gz')
        mask_path = os.path.join(original_images_paths, patient_name, 'mask.nii.gz')
        mae, ssim, psnr = evaluate_patient(pred_path, og_image_path, mask_path)
        patient_metrics.append([patient_name, mae, ssim, psnr])

    # Save the metDrics in a c.s.v file
    with open(f'{exp_name}_patient_metrics.csv', 'w') as f:
        for patient in patient_metrics:
            f.write(f'{patient[0]},{patient[1]},{patient[2]},{patient[3]}\n')

    return patient_metrics


if __name__ == '__main__':
    # Define the paths
    predictions_path = f'/Users/alexboving/Documents/GitHub/synthrad/data/Task1/brain_training_test_eval'
    original_images_path = f'/Users/alexboving/Documents/GitHub/synthrad/data/Task1/predictions'

    # Evaluate the patients
    patients_metrics_dict = evaluate_all_patients(predictions_path, original_images_path, exp_name='brain_training_test_eval')

