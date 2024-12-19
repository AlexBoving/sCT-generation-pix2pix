''' This script is used to prepare the prediction .png files, turn them into .nii.gz files so they
can be evaluated using the SynthSeg evaluation script. '''

import os
import numpy as np
import nibabel as nib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def get_patients_png_images_dict(predictions_path, original_images_path, prediction_string='fake_B'):
    ''' This function gets the list of the .png files of the predictions directory divides them by patient
    and returns a dictionary with the patient name as key and the list of the .png file paths as value. '''

    # Get the list of the .png files
    png_files_list = os.listdir(predictions_path)

    # Get a list of the name of the patients in the predictions directory
    patients_list = os.listdir(original_images_path)

    # Create a dictionary with the patient name as key and the list of the .png file paths as value
    patients_png_images_dict = {}
    
    # remove the .DS_Store file from the list
    if '.DS_Store' in patients_list:
        patients_list.remove('.DS_Store')
    elif '.DS_Store' in png_files_list:
        png_files_list.remove('.DS_Store')

    for patient in patients_list:
        patients_png_images_dict[patient] = []
        for png_file in png_files_list:
            if patient in png_file and prediction_string in png_file:
                patients_png_images_dict[patient].append(os.path.join(predictions_path, png_file))

        # Sort the list of the .png file paths
        patients_png_images_dict[patient].sort()

    return patients_png_images_dict

def expanse(predicted_image, original_image,  old_size=(256, 256)):
    '''
    Expand the image to the new size.
    '''

    # Pad the slice to the new size
    crop_left = (old_size[0] - original_image.height) // 2
    crop_top = (old_size[1] - original_image.width) // 2
    crop_right = old_size[0] - original_image.height - crop_left
    crop_bottom = old_size[1] - original_image.width - crop_top

    return ImageOps.expand(predicted_image, (-crop_left, -crop_top, -crop_right, -crop_bottom), fill=0)

def convert_rgb_to_grayscale(rgb_image):
    """
    Convert an 8-bit RGB image back to a 16-bit grayscale image
    Shift the red channel to its original position (12th to 9th bits)
    Shift the green channel to its original position (8th to 5th bits)
    The blue channel is already in the correct position (4th to 1st bits)
    Going back from the bit-placement method.
    """

    R = (rgb_image[:, :, 0] >> 4).astype(np.uint16)  # Extract the 4 most significant bits
    G = (rgb_image[:, :, 1] >> 4).astype(np.uint16)  # Extract the 4 most significant bits
    B = (rgb_image[:, :, 2] >> 4).astype(np.uint16)  # Extract the 4 most significant bits

    return (R << 8) | (G << 4) | B

    """
    # Split the RGB image into color channels
    # Extract the 5 most significant bits from each 8-bit channel
    Going back from the without bit-placement method.
    R5 = (rgb_image[:, :, 0] >> 3).astype(np.uint16)
    G5 = (rgb_image[:, :, 1] >> 3).astype(np.uint16)
    B5 = (rgb_image[:, :, 2] >> 2).astype(np.uint16)

    # Combine the 5-bit values into a single 16-bit grayscale value
    return (R5 << 11) | (G5 << 6) | B5
    """

def patient_png_to_nii_gz(patient_name, patient_png_list, original_images_path, result_path, mode='ct', rgb=False, _8bit=False):
    ''' This function uses the .png files of a patient, turns them into a 3D array
    and turns them into a .nii.gz file taking into account the information from the original .nii.gz.'''

    # Get the original .nii.gz file
    original_nifti_path = os.path.join(original_images_path, patient_name, f'{mode}.nii.gz')
    original_mask_path = os.path.join(original_images_path, patient_name, 'mask.nii.gz')

    # Get the original .nii.gz file information
    original_nifti = nib.load(original_nifti_path)
    original_nifti_array = original_nifti.get_fdata()
    original_nifti_affine = original_nifti.affine
    original_nifti_header = original_nifti.header

    # Get the original mask information
    original_mask = nib.load(original_mask_path)
    original_mask_array = original_mask.get_fdata()

    # Create the 3D array with the .png files based on the original .nii.gz file shape
    array_shape = original_nifti_array.shape
    patient_png_array = np.zeros(array_shape, dtype=np.float32) # Before it was float32
    for i, png_file_path in enumerate(patient_png_list):
        print(f'Processing slice {i} of patient {patient_name}')

        if rgb == True:
            image = Image.open(png_file_path).convert('RGB')
        elif _8bit == True:
            image = Image.open(png_file_path).convert('L')
        else:
            image = Image.open(png_file_path).convert('I')

        # Flip the image horizontally.
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Expand the image to the original size.
        image = expanse(image, Image.fromarray(original_nifti_array[:, :, i]))

        # convert the image to a numpy array
        image_array = np.array(image, dtype=np.float32)

        if rgb == True:
            # Convert the 8bit RGB image to a 16bit grayscale image
            image_array = convert_rgb_to_grayscale(image_array)

        image_array = image_array.astype(np.float32)
        
        # Rotate the image 90 degrees
        image_array = np.rot90(image_array)

        # Apply mask to the image_array
        image_array = np.where(original_mask_array[:, :, i] == 0, 0, image_array)

        if _8bit == True:
            image_array = image_array / 255.0
            image_array = (np.max(original_nifti_array[:, :, i]) - np.min(original_nifti_array[:, :, i])) * image_array + np.min(original_nifti_array[:, :, i])
        elif rgb == False:
            image_array = image_array / 65535.0
            image_array = (np.max(original_nifti_array[:, :, i]) - np.min(original_nifti_array[:, :, i])) * image_array + np.min(original_nifti_array[:, :, i])
        else:
            image_array -= np.abs(np.min(original_nifti_array[:, :, i]))

        patient_png_array[:, :, i] = image_array

        #Check the mean and min and max values of the image_array and print them
        print(f'\n\nPatient {patient_name} slice {i} mean: {np.mean(image_array)}')
        print(f'Patient {patient_name} slice {i} min: {np.min(image_array)}')
        print(f'Patient {patient_name} slice {i} max: {np.max(image_array)}')

        # Print the mean, max and min of the original nifti array slice i
        print(f'\nPatient {patient_name} slice {i} mean: {np.mean(original_nifti_array[:, :, i])}')
        print(f'Patient {patient_name} slice {i} min: {np.min(original_nifti_array[:, :, i])}')
        print(f'Patient {patient_name} slice {i} max: {np.max(original_nifti_array[:, :, i])}')

        """
        # Plot the png_image and the original_nifti_image
        if(i == 112 and patient_name == '1BA001'):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(png_array, cmap='gray')
            axs[0].set_title('synthetic CT')
            axs[1].imshow(original_nifti_array[:, :, i], cmap='gray')
            axs[1].set_title('original CT')
            axs[2].imshow(np.abs(original_nifti_array[:, :, i] - png_array), cmap='gray')
            axs[2].set_title('Absolute Difference')
            plt.show()
        """

    # Print the min and max values of the patient_png_array
    print(f'Patient {patient_name} min: {np.min(patient_png_array)}, max: {np.max(patient_png_array)}')
    # Print the min and max values of the original_nifti_array
    print(f'Patient {patient_name} min: {np.min(original_nifti_array)}, max: {np.max(original_nifti_array)}')

    #Create the .nii.gz file
    patient_nifti = nib.Nifti1Image(patient_png_array, original_nifti_affine, original_nifti_header)

    # Save the .nii.gz file
    result_nifti_path = os.path.join(result_path, f'{patient_name}_{mode}_prediction.nii.gz')
    nib.save(patient_nifti, result_nifti_path)

def patients_predictions_to_nifti(predictions_path, original_images_path, results_path, rgb=False, _8bit=False):

    # Get the list of the .png files of the predictions directory divides them by patient
    patients_png_images_dict = get_patients_png_images_dict(predictions_path, original_images_path)

    # Create the result directory
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # For each patient, turn the .png files into a .nii.gz file
    for patient_name, patient_png_list in patients_png_images_dict.items():
        patient_png_to_nii_gz(patient_name, patient_png_list, original_images_path, results_path, mode='ct', rgb=rgb, _8bit=_8bit)

"""
# Define the paths
predictions_path = f'/Users/alexboving/Desktop/Thèse/synthrad-pix2pix/Testing/pix2pix/fake_B'
original_images_path = f'/Users/alexboving/Desktop/Thèse/synthrad-pix2pix/Testing/Task1_16bit_5stacked/original'
results_path = f'/Users/alexboving/Desktop/Thèse/synthrad-pix2pix/Testing/Task1_16bit_5stacked/brain_training_test_eval'

# Turn the predictions into .nii.gz files
patients_predictions_to_nifti(predictions_path, original_images_path, results_path, rgb=False, _8bit=False)
"""