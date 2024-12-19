'''
This script prepares and splits the dataset of the MICCAI 2023 Grand Challenge SynthRad2023.
'''
import os, sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def split_patients(original_dataset_path, train_size=0.80, val_size=0.10, save_csv=True):
    '''
    Splits the dataset into train, validation and test sets.
    '''
    patients = os.listdir(original_dataset_path)
    patients.sort()
    np.random.seed(42)
    np.random.shuffle(patients)
    num_patients = len(patients)

    train_split = int(np.floor(train_size * num_patients))
    val_split = int(np.floor((train_size + val_size) * num_patients))
    train_patients = patients[:train_split]
    val_patients = patients[train_split:val_split]
    test_patients = patients[val_split:]

    # Save the splits in a csv file
    if save_csv:
        df = pd.DataFrame({'train': train_patients, 'val': val_patients, 'test': test_patients})
        df.to_csv(os.path.join(original_dataset_path, 'split.csv'), index=False)
    return train_patients, val_patients, test_patients


def patient_nifti_to_png(nifti_path, patient_name, result_path):
    '''
    Turns the patient NIfTI file of an image modality into numbered 16-bit grayscale PNG files.
    '''
    # Load the NIfTI file
    nii_img = nib.load(nifti_path)
    nii_array = nii_img.get_fdata()

    nii_array = nii_array.transpose()

    for idx, img_slice in enumerate(nii_array):
        # Add the minimum value to the slice to make it positive
        img_slice += abs(img_slice.min())

        #print(f'Minimum value: {img_slice.min()}')
        #print(f'Maximum value: {img_slice.max()}')

        img_slice[img_slice[0, 0] == img_slice]
        # Pad the slice to the new size
        img_slice = padding(img_slice, new_size=(256, 256))
        # Convert the numpy array to a Pillow Image
        img_slice = Image.fromarray(img_slice).convert('I;16')
        # Plot the image
        if idx == 110:
            plt.imshow(img_slice, cmap='gray')
            plt.show()
        # Save it as PNG
        img_slice.save(os.path.join(result_path, f'{patient_name}_{idx:03d}.png'))

def split_to_png(original_dataset_path, result_dataset_path, modality='ct'):
    '''
    Splits the dataset into train, validation and test sets and turns the nifti files into png files.
    '''

    # Split the dataset
    train_patients, val_patients, test_patients = split_patients(original_dataset_path, save_csv=False)

    # Create the folders for the png files
    train_path = os.path.join(result_dataset_path, 'train')
    val_path = os.path.join(result_dataset_path, 'val')
    test_path = os.path.join(result_dataset_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Turn the nifti files into png files
    for patient in train_patients:
        if patient == '.DS_Store':
            continue
        # Prints the number of the patient and the total of patients being processed
        print(f'Patient {train_patients.index(patient)+1}/{len(train_patients)}')
        patient_path = os.path.join(original_dataset_path, patient)
        patient_nifti_to_png(os.path.join(patient_path, f'{modality}.nii.gz'), patient, train_path)
    for patient in val_patients:
        if patient == '.DS_Store':
            continue
        # Prints the number of the patient and the total of patients being processed
        print(f'Patient {val_patients.index(patient)+1}/{len(val_patients)}')
        patient_path = os.path.join(original_dataset_path, patient)
        patient_nifti_to_png(os.path.join(patient_path, f'{modality}.nii.gz'), patient, val_path)
    for patient in test_patients:
        if patient == '.DS_Store':
            continue
        # Prints the number of the patient and the total of patients being processed
        print(f'Patient {test_patients.index(patient)+1}/{len(test_patients)}')
        patient_path = os.path.join(original_dataset_path, patient)
        patient_nifti_to_png(os.path.join(patient_path, f'{modality}.nii.gz'),patient, test_path)

def padding(image, new_size):
    '''
    Pads the image to the new size.
    '''
    # Convert the image to a PIL image
    image_slice = Image.fromarray(image)
    # Pad the slice to the new size
    pad_left = (new_size[0] - image_slice.width) // 2
    pad_top = (new_size[1] - image_slice.height) // 2
    pad_right = new_size[0] - image_slice.width - pad_left
    pad_bottom = new_size[1] - image_slice.height - pad_top
    # Apply padding
    padded_img = ImageOps.expand(image_slice, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

    # convert the image back to a numpy array
    padded_img = np.array(padded_img).astype(np.uint16)
    return padded_img

if __name__ == '__main__':
    task = 'Task1'
    modality = 'ct'
    structure = 'brain'
    subset = ''
    if len(sys.argv) > 1:
        original_dataset_path = sys.argv[1]
        result_dataset_path = sys.argv[2]
    else:
        original_dataset_path = f'/Users/alexboving/Desktop/Thèse/synthrad-pix2pix/{task}/{structure}'
        result_dataset_path = f'/Users/alexboving/Desktop/Thèse/synthrad-pix2pix/{task}_16bit_gray/{structure}_{modality}/'
    split_to_png(original_dataset_path, result_dataset_path, modality=modality)