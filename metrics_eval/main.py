import os
import shutil
from prediction_preparation import patients_predictions_to_nifti
from evaluation_2D import evaluate_all_patients as eval_2D
from evaluation_3D import evaluate_all_patients as eval_3D

def separate(directory):
    # Define the destination directories for real and fake images
    real_dir = os.path.join(directory, "real_B")
    fake_dir = os.path.join(directory, "fake_B")
    real_dir_A = os.path.join(directory, "real_A")

    # Create the destination directories if they don't exist
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(real_dir_A, exist_ok=True)

    # Iterate over the files in the source directory
    for filename in os.listdir(directory):
        if "_real_B" in filename:
            # Move real CT images to the real_dir
            shutil.move(os.path.join(directory, filename), os.path.join(real_dir, filename))
        elif "_fake_B" in filename:
            # Move fake CT images to the fake_dir
            shutil.move(os.path.join(directory, filename), os.path.join(fake_dir, filename))
        elif "_real_A" in filename:
            # Move real MRI images to the real_dir_A
            shutil.move(os.path.join(directory, filename), os.path.join(real_dir_A, filename))

    print("Files have been separated successfully.")

    # Return the fake_dir
    return fake_dir

if __name__ == '__main__':
    # Define the source directory containing the images.
    fake_images_path = separate('images') # Organize the images into real and fake directories and return the fake directory.
    directory = 'Task1'

    # Turn the predictions into .nii.gz files.
    patients_predictions_to_nifti(fake_images_path, f'{directory}/original', f'{directory}/brain_training_test_eval', rgb=False, _8bit=False)

    # Evaluate the predictions.
    eval_2D(f'{directory}/brain_training_test_eval', f'{directory}/original', directory)
    eval_3D(f'{directory}/brain_training_test_eval', f'{directory}/original', directory)