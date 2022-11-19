import os
from PIL import Image

import pandas as pd

import numpy as np

import time

from src.enc.utility import concat_image

class JpegAccelIR():
    def __init__(self, profiles_path, num_importance):
        self.size_qual_stats = pd.read_csv(profiles_path)
        self.num_importance = num_importance

        self.quality_candi = [10, 20, 30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95]

        # estimated size per quaity per importance
        self.estimated_size_info = self._get_estimated_size_info()
        # estimated size per to-quality per importance
        self.estimated_slope_info = self._get_estimated_slope_info()
        self.arrival_slope_info = self._get_arrival_slope_info()

    def _get_estimated_size_info(self):
        estimated_size_info = []
        num_importance = self.num_importance
        for importance in range(num_importance):
            importance_df = self.size_qual_stats[(self.size_qual_stats['importance']) == importance]
            size_per_qual = {}
            for q in self.quality_candi:
                qual_df = importance_df[(importance_df['quality'] == q)]
                size = qual_df.groupby(by=['quality']).mean()['size'].iloc[0]
                size_per_qual[q] = size
            estimated_size_info += [size_per_qual]
        return estimated_size_info
    
    def _get_estimated_slope_info(self):
        estimated_slope_info = []
        num_importance = self.num_importance
        for importance in range(num_importance):
            importance_df = self.size_qual_stats[(self.size_qual_stats['importance']) == importance]
            slope_per_qual = {}
            grouped = importance_df.groupby(by=['quality'])
            mean_info = grouped.mean()
            for i in range(len(self.quality_candi)-1):
                base_qual = self.quality_candi[i]
                lead_qual = self.quality_candi[i+1]
                base_size = mean_info.loc[base_qual, 'size']
                lead_size = mean_info.loc[lead_qual, 'size']
                base_l2 = mean_info.loc[base_qual, 'l2']
                lead_l2 = mean_info.loc[lead_qual, 'l2']
                slope = (lead_l2 - base_l2) / (lead_size - base_size)
                slope_per_qual[lead_qual] = slope
            estimated_slope_info += [slope_per_qual]
        return estimated_slope_info

    def _get_arrival_slope_info(self):
        arrival_slope_info = []
        num_importance = self.num_importance
        for importance in range(num_importance):
            importance_df = self.size_qual_stats[(self.size_qual_stats['importance']) == importance]
            slope_per_qual = {}
            grouped = importance_df.groupby(by=['quality'])
            mean_info = grouped.mean()
            for i in range(len(self.quality_candi)-1):
                base_qual = self.quality_candi[i]
                lead_qual = self.quality_candi[-1]
                base_size = mean_info.loc[base_qual, 'size']
                lead_size = mean_info.loc[lead_qual, 'size']
                base_l2 = mean_info.loc[base_qual, 'l2']
                lead_l2 = mean_info.loc[lead_qual, 'l2']
                slope = (lead_l2 - base_l2) / (lead_size - base_size)
                slope_per_qual[base_qual] = slope
            arrival_slope_info += [slope_per_qual]
        return arrival_slope_info

    def _init_importance_info(self, np_profile):
        importance_info = []
        for i in range(self.num_importance):
            info = []
            population = np_profile[np.where(np_profile[:, 3] == i)]
            info += [len(population)]
            info += [self.quality_candi[0]]
            importance_info.append(info)
        return importance_info

    def _init_importance_slope_info(self, importance_info, importance_candi):
        init_importance_slope_info = {}
        for importance in importance_candi:
            quality = importance_info[importance][1]
            importance_slope_info = self.estimated_slope_info[importance]
            slope = importance_slope_info[self.quality_candi[self.quality_candi.index(quality)+1]]
            init_importance_slope_info[importance] = slope
        return init_importance_slope_info
    
    def _init_importance_a_slope_info(self, importance_info, importance_candi, lam):
        init_importance_slope_info = {}
        for importance in importance_candi:
            quality = importance_info[importance][1]
            importance_slope_info = self.estimated_slope_info[importance]
            slope = importance_slope_info[self.quality_candi[self.quality_candi.index(quality)+1]]
            arrival_slope_info = self.arrival_slope_info[importance]
            arrival_slope = arrival_slope_info[self.quality_candi[self.quality_candi.index(quality)]]
            slope = slope + arrival_slope * lam
            init_importance_slope_info[importance] = slope
        return init_importance_slope_info

    def enc(self, image, profile, tmp_path, target_size, lam):
        # load profile
        np_profile = np.array(profile)

        # get unique importance
        importance_candi = sorted(np.unique(np_profile[:, 3]))

        importance_info = self._init_importance_info(np_profile)
        importance_slope_info = self._init_importance_a_slope_info(importance_info, importance_candi, lam)

        # get target size, base_size
        base_size = 0
        for importance in importance_candi:
            size_info = self.estimated_size_info[importance]
            info = importance_info[importance]
            num = info[0]
            base_size_info = size_info[self.quality_candi[0]]
            base_importance_size = num * base_size_info
            base_size += base_importance_size

        # greedy search
        update_seq = []
        start_time = time.time()
        while (base_size <= target_size):
            interrupt_signal = True
            for slope in importance_slope_info.values():
                if slope != 10000:
                    interrupt_signal = False
            
            if interrupt_signal:
                break

            best_importance = sorted(importance_slope_info.items(), key=lambda x: x[1])[0][0]
            # update info
            before_size = self.estimated_size_info[best_importance][importance_info[best_importance][1]]
            importance_info[best_importance][1] = self.quality_candi[self.quality_candi.index(importance_info[best_importance][1]) + 1]
            if importance_info[best_importance][1] == 95:
                importance_slope_info[best_importance] = 10000
            else:
                importance_slope_info[best_importance] = self.estimated_slope_info[best_importance][self.quality_candi[self.quality_candi.index(importance_info[best_importance][1]) + 1]] + self.arrival_slope_info[best_importance][self.quality_candi[self.quality_candi.index(importance_info[best_importance][1])]] * lam
            # update size
            after_size = self.estimated_size_info[best_importance][importance_info[best_importance][1]]
            delta_size = (after_size - before_size) * importance_info[best_importance][0]
            base_size += delta_size
            update_seq += [best_importance]
        end_time = time.time()
        elapsed_time = end_time - start_time

        # generate profile
        for idx in range(len(profile)):
            patch_importance = profile[idx][3]
            profile[idx].append(importance_info[patch_importance][1])
             
        # do encoding
        patch_list = []
        patch_size_list = []
        patch_quality_list = []
        enc_size = 0
        for patch_profile in profile:
            patch_width = patch_profile[0]
            patch_height = patch_profile[1]
            patch_size = patch_profile[2]
            patch_importance = patch_profile[3]
            if patch_width + patch_size > image.width and patch_height + patch_size <= image.height:
                patch_image = image.crop((patch_width, patch_height, image.width, patch_height+patch_size))
            elif patch_width + patch_size <= image.width and patch_height + patch_size > image.height:
                patch_image = image.crop((patch_width, patch_height, patch_width+patch_size, image.height))
            elif patch_width + patch_size > image.width and patch_height + patch_size > image.height:
                patch_image = image.crop((patch_width, patch_height, image.width, image.height))
            else:
                patch_image = image.crop((patch_width, patch_height, patch_width+patch_size, patch_height+patch_size))
            patch_image.save(tmp_path, quality=importance_info[patch_importance][1])
            enc_patch_image = Image.open(tmp_path)
            enc_patch_image.load()
            patch_size = os.path.getsize(tmp_path) - 625
            os.remove(tmp_path)
            enc_size += patch_size
            patch_list += [enc_patch_image]
            patch_size_list += [patch_size]
            patch_quality_list += [importance_info[patch_importance][1]]
        
        enc_image = concat_image(patch_list, patch_profile[2], image.width, image.height)
        enc_image.load()
        
        return enc_image, enc_size
    
    def generate_quality_map(self, image, profile, target_size):
        # load profile
        np_profile = np.array(profile)

        # get unique importance
        importance_candi = sorted(np.unique(np_profile[:, 3]))

        importance_info = self._init_importance_info(np_profile)
        importance_slope_info = self._init_importance_slope_info(importance_info, importance_candi)

        # get target size, base_size
        base_size = 0
        for importance in importance_candi:
            size_info = self.estimated_size_info[importance]
            info = importance_info[importance]
            num = info[0]
            base_size_info = size_info[self.quality_candi[0]]
            base_importance_size = num * base_size_info
            base_size += base_importance_size

        # greedy search
        update_seq = []
        start_time = time.time()
        while (base_size <= target_size):
            interrupt_signal = True
            for slope in importance_slope_info.values():
                if slope != 10000:
                    interrupt_signal = False
            
            if interrupt_signal:
                break

            best_importance = sorted(importance_slope_info.items(), key=lambda x: x[1])[0][0]
            # update info
            before_size = self.estimated_size_info[best_importance][importance_info[best_importance][1]]
            importance_info[best_importance][1] = self.quality_candi[self.quality_candi.index(importance_info[best_importance][1]) + 1]
            if importance_info[best_importance][1] == 95:
                importance_slope_info[best_importance] = 10000
            else:
                importance_slope_info[best_importance] = self.estimated_slope_info[best_importance][self.quality_candi[self.quality_candi.index(importance_info[best_importance][1]) + 1]]
            # update size
            after_size = self.estimated_size_info[best_importance][importance_info[best_importance][1]]
            delta_size = (after_size - before_size) * importance_info[best_importance][0]
            base_size += delta_size
            update_seq += [best_importance]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Consumed time for greedy search: {elapsed_time}")

        # generate profile
        for idx in range(len(profile)):
            patch_importance = profile[idx][3]
            profile[idx].append(importance_info[patch_importance][1])
             
        patch_quality_list = []
        for patch_profile in profile:
            patch_quality_list += [importance_info[patch_importance][1]]
        
        # make quality map
        quality_map = np.zeros((image.height, image.width))
        for i, patch_profile in enumerate(profile):
            patch_width = patch_profile[0]
            patch_height = patch_profile[1]
            patch_size = patch_profile[2]
            patch_importance = patch_profile[3]
            if patch_width + patch_size > image.width and patch_height + patch_size <= image.height:
                quality_map[patch_height: patch_height + patch_size, patch_width: image.width] += patch_quality_list[i]
            elif patch_width + patch_size <= image.width and patch_height + patch_size > image.height:
                quality_map[patch_height: image.height, patch_width: patch_width + patch_size] += patch_quality_list[i]
            elif patch_width + patch_size > image.width and patch_height + patch_size > image.height:
                quality_map[patch_height: image.height, patch_width: image.width] += patch_quality_list[i]
            else:
                quality_map[patch_height: patch_height + patch_size, patch_width: patch_width + patch_size] += patch_quality_list[i]

        return quality_map
