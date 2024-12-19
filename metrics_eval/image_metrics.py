#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop
from scipy import ndimage


def transform_image_resolution(image, new_resolution):
    current_resolution = image.shape[1:]  # Exclude channel dimension
    scale_factors = (1, new_resolution[0] / current_resolution[0], new_resolution[1] / current_resolution[1])

    transformed_image = ndimage.zoom(image, scale_factors)
    return transformed_image

class ImageMetrics():
    def __init__(self):
        # Use fixed wide dynamic range
        self.dynamic_range = [-1024., 3000.]

    def score_patient(self, ground_truth, predicted, mask):
        """
        gt = SimpleITK.ReadImage(ground_truth_path)
        pred = SimpleITK.ReadImage(predicted_path)
        mask = SimpleITK.ReadImage(mask_path)

        caster = SimpleITK.CastImageFilter()
        caster.SetOutputPixelType(SimpleITK.sitkFloat32)
        caster.SetNumberOfThreads(1)

        gt = caster.Execute(gt)
        pred = caster.Execute(pred)
        mask = caster.Execute(mask)

        # Get numpy array from SITK Image
        gt_array = SimpleITK.GetArrayFromImage(gt)
        pred_array = SimpleITK.GetArrayFromImage(pred)
        mask_array = SimpleITK.GetArrayFromImage(mask)
        """
        gt_array = ground_truth
        pred_array = predicted
        mask_array = mask

        # Plot the prediction and the groundtruth images slices 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        # for i in range(10, 110, 10):
        #     # Plot the groundtruth image
        #     gt_slice = gt_array[i, :, :]
        #     gt_slice = np.clip(gt_slice, min(self.dynamic_range), max(self.dynamic_range))
        #     gt_slice = (gt_slice - min(self.dynamic_range)) / (max(self.dynamic_range) - min(self.dynamic_range))
        #     gt_slice = np.uint8(gt_slice * 255)
        #     # Plot the prediction image
        #     pred_slice = pred_array[i, :, :]
        #     pred_slice = np.clip(pred_slice, min(self.dynamic_range), max(self.dynamic_range))
        #     pred_slice = (pred_slice - min(self.dynamic_range)) / (max(self.dynamic_range) - min(self.dynamic_range))
        #     pred_slice = np.uint8(pred_slice * 255)
        #     # Plot the mask image
        #     mask_slice = mask_array[i, :, :]
        #     mask_slice = np.clip(mask_slice, min(self.dynamic_range), max(self.dynamic_range))
        #     mask_slice = (mask_slice - min(self.dynamic_range)) / (max(self.dynamic_range) - min(self.dynamic_range))
        #     mask_slice = np.uint8(mask_slice * 255)
        #     # Save the images
        #     gt_slice = Image.fromarray(gt_slice)
        #     pred_slice = Image.fromarray(pred_slice)
        #     mask_slice = Image.fromarray(mask_slice)
        #     gt_slice.save(f'gt_slice_{i}.png')
        #     pred_slice.save(f'pred_slice_{i}.png')
        #     mask_slice.save(f'mask_slice_{i}.png')

        # Calculate image metrics
        mse_value = self.mse(gt_array,
                             pred_array,
                             mask_array)
        
        mae_value = self.mae(gt_array,
                             pred_array,
                             mask_array)

        psnr_value = self.psnr(gt_array,
                               pred_array,
                               mask_array,
                               use_population_range=True)

        ssim_value = self.ssim(gt_array,
                               pred_array,
                               mask_array)
        return {
            'mse': mse_value,
            'mae': mae_value,
            'ssim': ssim_value,
            'psnr': psnr_value
        }
    
    def mse(self,
            gt: np.ndarray,
            pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Mean Squared Error (MSE)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).

        Returns
        -------
        mae : float
            mean absolute error.

        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)
        
        mse_value = np.sum(np.square(gt * mask - pred * mask)) / mask.sum()
        return float(mse_value)


    def mae(self,
            gt: np.ndarray,
            pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Mean Absolute Error (MAE)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).

        Returns
        -------
        mae : float
            mean absolute error.

        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)

        mae_value = np.sum(np.abs(gt * mask - pred * mask)) / mask.sum()
        return float(mae_value)

    def psnr(self,
             gt: np.ndarray,
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None,
             use_population_range: Optional[bool] = False) -> float:
        """
        Compute Peak Signal to Noise Ratio metric (PSNR)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
        use_population_range : bool, optional
            When a predefined population wide dynamic range should be used.
            gt and pred will also be clipped to these values.

        Returns
        -------
        psnr : float
            Peak signal to noise ratio..

        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)

        if use_population_range:
            dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]

            # Clip gt and pred to the dynamic range
            gt = np.where(gt < self.dynamic_range[0], self.dynamic_range[0], gt)
            gt = np.where(gt > self.dynamic_range[1], self.dynamic_range[1], gt)
            pred = np.where(pred < self.dynamic_range[0], self.dynamic_range[0], pred)
            pred = np.where(pred > self.dynamic_range[1], self.dynamic_range[1], pred)
        else:
            dynamic_range = gt.max() - gt.min()

        # apply mask
        gt = gt[mask == 1]
        pred = pred[mask == 1]
        psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range)
        return float(psnr_value)

    def ssim(self,
             gt: np.ndarray,
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Structural Similarity Index Metric (SSIM)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).

        Returns
        -------
        ssim : float
            structural similarity index metric.

        """
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        if mask is not None:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)

            # Mask gt and pred
            gt = np.where(mask == 0, min(self.dynamic_range), gt)
            pred = np.where(mask == 0, min(self.dynamic_range), pred)

        # Make values non-negative
        if min(self.dynamic_range) < 0:
            gt = gt - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        # Set dynamic range for ssim calculation and calculate ssim_map per pixel
        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]
        ssim_value_full, ssim_map = structural_similarity(gt, pred, data_range=dynamic_range, full=True)

        if mask is not None:
            # Follow skimage implementation of calculating the mean value:
            # crop(ssim_map, pad).mean(dtype=np.float64), with pad=3 by default.
            pad = 3
            ssim_value_masked = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)
            return ssim_value_masked
        else:
            return ssim_value_full


if __name__ == '__main__':
    metrics = ImageMetrics()
    ground_truth_path = "path/to/ground_truth.mha"
    predicted_path = "path/to/prediction.mha"
    mask_path = "path/to/mask.mha"
    print(metrics.score_patient(ground_truth_path, predicted_path, mask_path))