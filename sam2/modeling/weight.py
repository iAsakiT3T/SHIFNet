import numpy as np
import torch
from tqdm import tqdm
import os
import math
import random
import time
import torch.backends.cudnn as cudnn



class ClassWeight(object):

    def __init__(self, method):
        assert method in ['no', 'enet', 'median_freq_balancing']
        self.method = method

    def get_weight(self, dataloader, num_classes):
        if self.method == 'no':
            return np.ones(num_classes)
        if self.method == 'enet':
            return self._enet_weighing(dataloader, num_classes)
        if self.method == 'median_freq_balancing':
            return self._median_freq_balancing(dataloader, num_classes)

    def _enet_weighing(self, dataloader, num_classes, c=1.02):
        """Computes class weights as described in the ENet paper:

            w_class = 1 / (ln(c + p_class)),

        where c is usually 1.02 and p_class is the propensity score of that
        class:

            propensity_score = freq_class / total_pixels.

        References: https://arxiv.org/abs/1606.02147

        Keyword arguments:
        - dataloader (``data.Dataloader``): A data loader to iterate over the
        dataset.
        - num_classes (``int``): The number of classes.
        - c (``int``, optional): AN additional hyper-parameter which restricts
        the interval of values for the weights. Default: 1.02.

        """
        print('computing class weight .......................')
        class_count = 0
        total = 0
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            label = sample['label']
            label = label.cpu().numpy()

            # Flatten label
            flat_label = label.flatten()

            # Sum up the number of pixels of each class and the total pixel
            # counts for each label
            class_count += np.bincount(flat_label, minlength=num_classes)
            total += flat_label.size

        # Compute propensity score and then the weights for each class
        propensity_score = class_count / total
        class_weights = 1 / (np.log(c + propensity_score))

        return class_weights

    def _median_freq_balancing(self, dataloader, num_classes):
        """Computes class weights using median frequency balancing as described
        in https://arxiv.org/abs/1411.4734:

            w_class = median_freq / freq_class,

        where freq_class is the number of pixels of a given class divided by
        the total number of pixels in images where that class is present, and
        median_freq is the median of freq_class.

        Keyword arguments:
        - dataloader (``data.Dataloader``): A data loader to iterate over the
        dataset.
        whose weights are going to be computed.
        - num_classes (``int``): The number of classes

        """
        print('computing class weight .......................')
        class_count = 0
        total = 0
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            label = sample['label']
            label = label.cpu().numpy()

            # Flatten label
            flat_label = label.flatten()

            # Sum up the class frequencies
            bincount = np.bincount(flat_label, minlength=num_classes)

            # Create of mask of classes that exist in the label
            mask = bincount > 0
            # Multiply the mask by the pixel count. The resulting array has
            # one element for each class. The value is either 0 (if the class
            # does not exist in the label) or equal to the pixel count (if
            # the class exists in the label)
            total += mask * flat_label.size

            # Sum up the number of pixels found for each class
            class_count += bincount

        # Compute the frequency and its median
        freq = class_count / total
        med = np.median(freq)

        return med / freq