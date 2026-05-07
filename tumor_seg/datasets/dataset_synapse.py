import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom,rotate
from torch.utils.data import Dataset
from PIL import Image
import json


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        # print('image:', image.shape, image.max(), image.min())
        # print('label:', label.shape, label.max(), label.min())
        sample = {'image': image, 'label': label.long()}
        return sample



def random_rot_flip_all(img, lbl, ref_img, ref_lbl):
    """同步旋转 + 翻转，保证主图和参考图一致变换"""
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    
    img = np.rot90(img, k)
    lbl = np.rot90(lbl, k)
    ref_img = np.rot90(ref_img, k)
    ref_lbl = np.rot90(ref_lbl, k)
    
    img = np.flip(img, axis=axis).copy()
    lbl = np.flip(lbl, axis=axis).copy()
    ref_img = np.flip(ref_img, axis=axis).copy()
    ref_lbl = np.flip(ref_lbl, axis=axis).copy()
    
    return img, lbl, ref_img, ref_lbl


def random_rotate_all(img, lbl, ref_img, ref_lbl):
    """同步任意角度旋转"""
    angle = np.random.randint(-20, 20)
    
    img = rotate(img, angle, reshape=False, order=0, mode='nearest')
    lbl = rotate(lbl, angle, reshape=False, order=0, mode='nearest')
    ref_img = rotate(ref_img, angle, reshape=False, order=0, mode='nearest')
    ref_lbl = rotate(ref_lbl, angle, reshape=False, order=0, mode='nearest')
    
    return img, lbl, ref_img, ref_lbl


class RandomGenerator_ref(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        ref_image = sample['ref_image']
        ref_label = sample['ref_mask']

        # 同步数据增强
        if random.random() > 0.5:
            image, label, ref_image, ref_label = random_rot_flip_all(image, label, ref_image, ref_label)
        elif random.random() > 0.5:
            image, label, ref_image, ref_label = random_rotate_all(image, label, ref_image, ref_label)

        # Resize
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            zoom_factor_x = self.output_size[0] / x
            zoom_factor_y = self.output_size[1] / y
            image = zoom(image, (zoom_factor_x, zoom_factor_y), order=3)
            label = zoom(label, (zoom_factor_x, zoom_factor_y), order=0)
            ref_image = zoom(ref_image, (zoom_factor_x, zoom_factor_y), order=3)
            ref_label = zoom(ref_label, (zoom_factor_x, zoom_factor_y), order=0)

        # To Tensor
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).long()
        ref_image = torch.from_numpy(ref_image.astype(np.float32))
        ref_label = torch.from_numpy(ref_label.astype(np.float32)).long()

        return {
            'image': image,
            'label': label,
            'ref_image': ref_image,
            'ref_mask': ref_label
        }


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, mode, transform=None):
        self.transform = transform  # using transform in torch!
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        self.mode = mode
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, "CT")
        # mask_dir = os.path.join(self.root_dir, "Mask")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            # patient_mask_folder = os.path.join(mask_dir, patient_folder)
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                mask_path = ct_path.replace('CT', 'Mask')
                
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
        
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0

        x, y = image.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        image = zoom(image, (224 / x, 224 / y), order=3)  # why not 3?
        mask = zoom(mask, (224 / x, 224 / y), order=0)
        # if self.split == "train":
        #     slice_name = self.sample_list[idx].strip('\n')
        #     data_path = os.path.join(self.data_dir, slice_name+'.npz')
        #     data = np.load(data_path)
        #     image, label = data['image'], data['label']   #512,512  img 0-1 label 0-8
            
        # else:
        #     vol_name = self.sample_list[idx].strip('\n')
        #     filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
        #     data = h5py.File(filepath)
        #     image, label = data['image'][:], data['label'][:]
        folder_name = os.path.dirname(mask_path)
        dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        ref_mask_path = dict[folder_name]
        ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_image.shape
        ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        
        ref_mask = zoom(ref_mask, (224 / ref_x, 224 / ref_y), order=0)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
        sample = {'image': image, 'label': mask}
                  
        # sample = {'image': image, 'label': mask, 'ref_image': ref_image, 'ref_label': ref_mask}
        if self.transform:
            sample = self.transform(sample)
            #倒数第二个文件夹名作为case_name
        sample['ref_image'] = ref_image
        sample['ref_mask'] = ref_mask
        sample['case_name'] = os.path.basename(os.path.dirname(image_path))
        sample['image_path'] = image_path
        sample['orig_size'] = np.array([x, y])  # original H, W before zoom to 224
        #打印sample的所有key
        # print('sample:', sample.keys())
        return sample


class Dataset_sli2vol(Dataset):   #use pseudo label as reference mask, the input only includes original image and reference/pseudo mask
    def __init__(self, base_dir, mode, transform=None):
        self.transform = transform  # using transform in torch!
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        self.dataset = base_dir.split('/')[-2]
        self.mode = mode
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, "CT")
        # mask_dir = os.path.join(self.root_dir, "Mask")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            # patient_mask_folder = os.path.join(mask_dir, patient_folder)
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                mask_path = ct_path.replace('CT', 'Mask')
                
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
        
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0

        x, y = image.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        image = zoom(image, (224 / x, 224 / y), order=3)  # why not 3?
        mask = zoom(mask, (224 / x, 224 / y), order=0)
        # if self.split == "train":
        #     slice_name = self.sample_list[idx].strip('\n')
        #     data_path = os.path.join(self.data_dir, slice_name+'.npz')
        #     data = np.load(data_path)
        #     image, label = data['image'], data['label']   #512,512  img 0-1 label 0-8
            
        # else:
        #     vol_name = self.sample_list[idx].strip('\n')
        #     filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
        #     data = h5py.File(filepath)
        #     image, label = data['image'][:], data['label'][:]
        case_num, slice_num = mask_path.split('/')[-2:]
        ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        dict_path = os.path.join('./Sli2Vol/result', self.dataset+'_train', 'annotation_dict_'+self.mode+'.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(mask_path)
        
        if ref_mask.max() == 0:
            ref_mask_path = dict[folder_name]
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # folder_name = os.path.dirname(mask_path)
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        # ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        # ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        # ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        
        ref_mask = zoom(ref_mask, (224 / ref_x, 224 / ref_y), order=0)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
        sample = {'image': image, 'label': mask}
                  
        # sample = {'image': image, 'label': mask, 'ref_image': ref_image, 'ref_label': ref_mask}
        if self.transform:
            sample = self.transform(sample)
            #倒数第二个文件夹名作为case_name
        sample['ref_image'] = ref_mask
        sample['ref_mask'] = ref_mask
        sample['case_name'] = os.path.basename(os.path.dirname(image_path))
        sample['image_path'] = image_path
        #打印sample的所有key
        # print('sample:', sample.keys())
        return sample

class Dataset_vol2flow(Dataset):   #use pseudo label as reference mask, the input only includes original image and reference/pseudo mask
    def __init__(self, base_dir, mode, transform=None):
        self.transform = transform  # using transform in torch!
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        self.dataset = base_dir.split('/')[-2]
        self.mode = mode
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, "CT")
        # mask_dir = os.path.join(self.root_dir, "Mask")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            # patient_mask_folder = os.path.join(mask_dir, patient_folder)
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                mask_path = ct_path.replace('CT', 'Mask')
                
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
        
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0

        x, y = image.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        image = zoom(image, (224 / x, 224 / y), order=3)  # why not 3?
        mask = zoom(mask, (224 / x, 224 / y), order=0)
        # if self.split == "train":
        #     slice_name = self.sample_list[idx].strip('\n')
        #     data_path = os.path.join(self.data_dir, slice_name+'.npz')
        #     data = np.load(data_path)
        #     image, label = data['image'], data['label']   #512,512  img 0-1 label 0-8
            
        # else:
        #     vol_name = self.sample_list[idx].strip('\n')
        #     filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
        #     data = h5py.File(filepath)
        #     image, label = data['image'][:], data['label'][:]
        case_num, slice_num = mask_path.split('/')[-2:]
        # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        ref_mask_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse',self.dataset+'_result',self.dataset, self.mode, 'Mask',case_num,slice_num)
        
        dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+self.mode+'.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(mask_path)
        #如果ref_mask_path不存在，则从json文件中读取
        if not os.path.exists(ref_mask_path):
            ref_mask_path = dict[folder_name]
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        # ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        # ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        #如果全0，则从json文件中读取
        if ref_mask.max() == 0:
            ref_mask_path = dict[folder_name]
            ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        
        ref_mask = zoom(ref_mask, (224 / ref_x, 224 / ref_y), order=0)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
        sample = {'image': image, 'label': mask}
                  
        # sample = {'image': image, 'label': mask, 'ref_image': ref_image, 'ref_label': ref_mask}
        if self.transform:
            sample = self.transform(sample)
            #倒数第二个文件夹名作为case_name
        sample['ref_image'] = ref_mask
        sample['ref_mask'] = ref_mask
        sample['case_name'] = os.path.basename(os.path.dirname(image_path))
        sample['image_path'] = image_path
        #打印sample的所有key
        # print('sample:', sample.keys())
        return sample
    

class Dataset_v5(Dataset):   #use largest slice as reference image
    def __init__(self, base_dir, mode, transform=None):
        self.transform = transform  # using transform in torch!
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        self.dataset = base_dir.split('/')[-2]
        self.mode = mode
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, "CT")
        # mask_dir = os.path.join(self.root_dir, "Mask")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            # patient_mask_folder = os.path.join(mask_dir, patient_folder)
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                mask_path = ct_path.replace('CT', 'Mask')
                
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
        
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0

        x, y = image.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        image = zoom(image, (224 / x, 224 / y), order=3)  # why not 3?
        mask = zoom(mask, (224 / x, 224 / y), order=0)
        # if self.split == "train":
        #     slice_name = self.sample_list[idx].strip('\n')
        #     data_path = os.path.join(self.data_dir, slice_name+'.npz')
        #     data = np.load(data_path)
        #     image, label = data['image'], data['label']   #512,512  img 0-1 label 0-8
            
        # else:
        #     vol_name = self.sample_list[idx].strip('\n')
        #     filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
        #     data = h5py.File(filepath)
        #     image, label = data['image'][:], data['label'][:]
        case_num, slice_num = mask_path.split('/')[-2:]
        # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        # ref_mask_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse',self.dataset+'_result',self.dataset, self.mode, 'Mask',case_num,slice_num)
        
        dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+self.mode+'.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(mask_path).replace('./','./')
        #如果ref_mask_path不存在，则从json文件中读取
        # if not os.path.exists(ref_mask_path):
        ref_mask_path = dict[folder_name].replace('./', './')
        ref_image_path = ref_mask_path.replace('Mask', 'CT')
        
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        # ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        
        ref_mask = zoom(ref_mask, (224 / ref_x, 224 / ref_y), order=0)
        sample = {
            'image': image,
            'label': mask,
            'ref_image': ref_image,
            'ref_mask': ref_mask
        }

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = os.path.basename(os.path.dirname(image_path))
        sample['image_path'] = image_path

        return sample
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
        # sample = {'image': image, 'label': mask}
                  
        # # sample = {'image': image, 'label': mask, 'ref_image': ref_image, 'ref_label': ref_mask}
        # if self.transform:
        #     sample = self.transform(sample)
        #     #倒数第二个文件夹名作为case_name
        # sample['ref_image'] = ref_image
        # sample['ref_mask'] = ref_mask
        # sample['case_name'] = os.path.basename(os.path.dirname(image_path))
        # sample['image_path'] = image_path
        # #打印sample的所有key
        # # print('sample:', sample.keys())
        # return sample

class Dataset_v5_2(Dataset):   # use middle slice as reference image
    def __init__(self, base_dir, mode, transform=None):
        self.transform = transform
        self.data_dir = base_dir
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        self.dataset = base_dir.split('/')[-2]
        self.mode = mode
        self.ref_map = {}
        dict_path = os.path.join(self.data_dir, "annotation_dict_middle.json")
        if os.path.exists(dict_path):
            print(f"[INFO] middle-slice json found: {dict_path}")
            with open(dict_path, "r") as f:
                self.ref_map = json.load(f)
        else:
            print(f"[WARN] middle-slice json not found: {dict_path}. Will fall back to per-sample self-reference.")

    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, "CT")
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            if not os.path.isdir(patient_ct_folder):
                continue
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                if not os.path.isfile(ct_path):
                    continue
                mask_path = ct_path.replace('CT', 'Mask')
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path  = self.mask_paths[idx]

        # ---- 原始加载 & 归一化/缩放（保持不变）----
        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        mask  = mask / 255.0
        x, y = image.shape
        image = zoom(image, (224 / x, 224 / y), order=3)
        mask  = zoom(mask,  (224 / x, 224 / y), order=0)

        # ========== [CHANGED] 选取 reference：用 middle-slice JSON ==========
        patient = os.path.basename(os.path.dirname(image_path))   # CT/<patient>/<file>
        ref_image_path, ref_mask_path = image_path, mask_path     # 默认回退：用当前切片自身

        info = self.ref_map.get(patient)
        if isinstance(info, dict):
            # 优先使用 JSON 里记录的绝对路径
            cand_ct  = info.get("ct_path")
            cand_msk = info.get("mask_path")  # 可能为 None
            if cand_ct and os.path.exists(cand_ct):
                ref_image_path = cand_ct
            else:
                print(f"[WARN] middle-slice not found: {cand_ct}. Will fall back to current slice.")
            # ref_mask_path 允许为 None（若该病人无任何 mask），这种情况回退到当前样本的 mask
            if cand_msk and os.path.exists(cand_msk):
                ref_mask_path = cand_msk
            else:
                print(f"[WARN] middle-slice not found: {cand_msk}. Will fall back to current slice.")
        # print('image_path:', image_path)
        # print('ref_image_path:', ref_image_path)
        # 加载 reference 并做同样预处理（保持不变）
        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        ref_mask  = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask  = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        ref_mask  = zoom(ref_mask,  (224 / ref_x, 224 / ref_y), order=0)

        sample = {
            'image': image,
            'label': mask,
            'ref_image': ref_image,
            'ref_mask': ref_mask
        }
        if self.transform:
            sample = self.transform(sample)
        sample['case_name']  = os.path.basename(os.path.dirname(image_path))
        sample['image_path'] = image_path
        sample['orig_size'] = np.array([x, y])  # original H, W before zoom to 224
        return sample




class Dataset_v5_1(Dataset):   #use largest slice (prediction from model1) as reference image
    def __init__(self, base_dir, mode, transform=None):
        self.transform = transform  # using transform in torch!
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        self.dataset = base_dir.split('/')[-2]
        self.mode = mode
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, "CT")
        # mask_dir = os.path.join(self.root_dir, "Mask")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            # patient_mask_folder = os.path.join(mask_dir, patient_folder)
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                mask_path = ct_path.replace('CT', 'Mask')
                
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
        
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0

        x, y = image.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        image = zoom(image, (224 / x, 224 / y), order=3)  # why not 3?
        mask = zoom(mask, (224 / x, 224 / y), order=0)
        # if self.split == "train":
        #     slice_name = self.sample_list[idx].strip('\n')
        #     data_path = os.path.join(self.data_dir, slice_name+'.npz')
        #     data = np.load(data_path)
        #     image, label = data['image'], data['label']   #512,512  img 0-1 label 0-8
            
        # else:
        #     vol_name = self.sample_list[idx].strip('\n')
        #     filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
        #     data = h5py.File(filepath)
        #     image, label = data['image'][:], data['label'][:]
        case_num, slice_num = mask_path.split('/')[-2:]
        # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        # ref_mask_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse',self.dataset+'_result',self.dataset, self.mode, 'Mask',case_num,slice_num)
        
        dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+self.mode+'.json')
        # dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+'train'+'.json')
        # print('dict_path:', dict_path)
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(mask_path).replace('./','./')
        #如果ref_mask_path不存在，则从json文件中读取
        # if not os.path.exists(ref_mask_path):
        ref_mask_path = dict[folder_name].replace('./', './')
        ref_image_path = ref_mask_path.replace('Mask', 'CT')
        ref_mask_path = ref_mask_path.replace('2D_data/local/test/Mask', 'project_TransUNet/my_predictions/TU_local224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224')
        # ./2D_data/local/test/Mask/Mening091/065.png
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # ./project_TransUNet/my_predictions/TU_local224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/Mening001
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        # ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
        ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x_m, ref_y_m = ref_mask.shape
        ref_x_i, ref_y_i = ref_image.shape
        ref_image = zoom(ref_image, (224 / ref_x_i, 224 / ref_y_i), order=3)
        
        ref_mask = zoom(ref_mask, (224 / ref_x_m, 224 / ref_y_m), order=0)
        sample = {
            'image': image,
            'label': mask,
            'ref_image': ref_image,
            'ref_mask': ref_mask
        }

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = os.path.basename(os.path.dirname(image_path))
        sample['image_path'] = image_path

        return sample
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
        # sample = {'image': image, 'label': mask}
                  
        # # sample = {'image': image, 'label': mask, 'ref_image': ref_image, 'ref_label': ref_mask}
        # if self.transform:
        #     sample = self.transform(sample)
        #     #倒数第二个文件夹名作为case_name
        # sample['ref_image'] = ref_image
        # sample['ref_mask'] = ref_mask
        # sample['case_name'] = os.path.basename(os.path.dirname(image_path))
        # sample['image_path'] = image_path
        # #打印sample的所有key
        # # print('sample:', sample.keys())
        # return sample


class Dataset_v6(Dataset):  #randomly choose another slice in the same volume as reference image
    def __init__(self, base_dir, mode, transform=None):
        self.transform = transform
        self.data_dir = base_dir
        self.mode = mode
        self.dataset = base_dir.split('/')[-2]
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        self.ref_dict = self._build_ref_dict()

    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, "CT")

        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                mask_path = ct_path.replace('CT', 'Mask')
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
        
        return image_paths, mask_paths

    def _build_ref_dict(self):
        ref_dict = {}
        for image_path in self.image_paths:
            folder = os.path.dirname(image_path)
            all_slices = sorted(os.listdir(folder))
            current_filename = os.path.basename(image_path)
            others = [os.path.join(folder, f) for f in all_slices if f != current_filename]
            ref_dict[image_path] = others
        return ref_dict

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        # print('image_path:', image_path)
        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0

        x, y = image.shape
        image = zoom(image, (224 / x, 224 / y), order=3)
        mask = zoom(mask, (224 / x, 224 / y), order=0)

        # 使用缓存好的 reference 列表
        ref_image_path = random.choice(self.ref_dict[image_path])
        ref_mask_path = ref_image_path.replace('CT', 'Mask')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)

        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0

        ref_x, ref_y = ref_mask.shape
        ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        ref_mask = zoom(ref_mask, (224 / ref_x, 224 / ref_y), order=0)
        sample = {
            'image': image,
            'label': mask,
            'ref_image': ref_image,
            'ref_mask': ref_mask
        }

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = os.path.basename(os.path.dirname(image_path))
        sample['image_path'] = image_path

        return sample
    
        
    

class Dataset_v6_2(Dataset):  # 5 fixed pairs per slice, total length = 5 * num_slices / n(n-1)
    def __init__(self, base_dir, mode, transform=None, k=5):
        self.transform = transform
        self.data_dir = base_dir
        self.mode = mode
        self.k = k  # number of references per slice

        self.dataset = base_dir.split('/')[-2]
        if self.k > 0:
            self.pairs = self._build_k_pairs()
        else:
            self.pairs = self._build_all_pairs()

    def _build_k_pairs(self):
        """构建所有的 (query, reference) 对"""
        pairs = []
        ct_root = os.path.join(self.data_dir, "CT")
        mask_root = os.path.join(self.data_dir, "Mask")

        for patient_folder in os.listdir(ct_root):
            ct_folder = os.path.join(ct_root, patient_folder)
            mask_folder = os.path.join(mask_root, patient_folder)
            slice_files = sorted(os.listdir(ct_folder))
            slice_paths = [
                (os.path.join(ct_folder, f), os.path.join(mask_folder, f))
                for f in slice_files if os.path.exists(os.path.join(mask_folder, f))
            ]

            for query_path, query_mask in slice_paths:
                # 取其他 slice 中的 k 个作为 reference（不含自己）
                other_refs = [p for p in slice_paths if p[0] != query_path]
                k_refs = random.sample(other_refs, min(self.k, len(other_refs)))
                for ref_path, ref_mask in k_refs:
                    pairs.append((query_path, query_mask, ref_path, ref_mask))

        return pairs
    
    def _build_all_pairs(self):
        """构建所有的 (query, reference) 对，ref 为同 volume 中其他所有 slice"""
        pairs = []
        ct_root = os.path.join(self.data_dir, "CT")
        mask_root = os.path.join(self.data_dir, "Mask")

        for patient_folder in os.listdir(ct_root):
            ct_folder = os.path.join(ct_root, patient_folder)
            mask_folder = os.path.join(mask_root, patient_folder)
            slice_files = sorted(os.listdir(ct_folder))
            slice_paths = [
                (os.path.join(ct_folder, f), os.path.join(mask_folder, f))
                for f in slice_files if os.path.exists(os.path.join(mask_folder, f))
            ]

            for query_path, query_mask in slice_paths:
                for ref_path, ref_mask in slice_paths:
                    if query_path != ref_path:
                        query_idx = int(os.path.basename(query_path).split('.')[0])
                        ref_idx = int(os.path.basename(ref_path).split('.')[0])
                        distance = query_idx - ref_idx
                        pairs.append((query_path, query_mask, ref_path, ref_mask, distance))

        return pairs


    def __len__(self):
        return len(self.pairs)

    def _load_and_process(self, img_path, mask_path):
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0

        h, w = image.shape
        image = zoom(image, (224 / h, 224 / w), order=3)
        mask = zoom(mask, (224 / h, 224 / w), order=0)

        # image = torch.tensor(image).unsqueeze(0).float()
        # mask = torch.tensor(mask).unsqueeze(0).float()
        return image, mask

    def __getitem__(self, idx):
        query_img_path, query_mask_path, ref_img_path, ref_mask_path, distance = self.pairs[idx]

        query_img, query_mask = self._load_and_process(query_img_path, query_mask_path)
        ref_img, ref_mask = self._load_and_process(ref_img_path, ref_mask_path)

        sample = {
            'image': query_img,
            'label': query_mask,
            'ref_image': ref_img,
            'ref_mask': ref_mask,
            # 'case_name': os.path.basename(os.path.dirname(query_img_path)),
            # 'image_path': query_img_path,
            # 'ref_path': ref_img_path
        }

        
        if self.transform:
            sample = self.transform(sample)
        sample['distance'] = distance
        sample['image_path'] = query_img_path

        return sample



class Dataset_v6_3(Dataset):  #Except the largest slice, also randomly choose another slice in the same volume as reference image
    def __init__(self, base_dir, mode, transform=None):
        self.transform = transform
        self.data_dir = base_dir
        self.mode = mode
        self.dataset = base_dir.split('/')[-2]
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        self.ref_dict = self._build_ref_dict()

    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, "CT")

        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                mask_path = ct_path.replace('CT', 'Mask')
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
        
        return image_paths, mask_paths

    def _build_ref_dict(self):
        ref_dict = {}
        for image_path in self.image_paths:
            folder = os.path.dirname(image_path)
            all_slices = sorted(os.listdir(folder))
            current_filename = os.path.basename(image_path)
            others = [os.path.join(folder, f) for f in all_slices if f != current_filename]
            ref_dict[image_path] = others
        return ref_dict

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        # print('image_path:', image_path)
        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0

        x, y = image.shape
        image = zoom(image, (224 / x, 224 / y), order=3)
        mask = zoom(mask, (224 / x, 224 / y), order=0)

        # 使用缓存好的 reference 列表
        ref_image_path1 = random.choice(self.ref_dict[image_path])
        ref_mask_path1 = ref_image_path1.replace('CT', 'Mask')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)

        ref_image1 = np.array(Image.open(ref_image_path1).convert("L"), dtype=np.float32)
        ref_mask1 = np.array(Image.open(ref_mask_path1).convert("L"), dtype=np.float32)
        ref_image1 = (ref_image1 - ref_image1.min()) / (ref_image1.max() - ref_image1.min())
        ref_mask1 = ref_mask1 / 255.0

        ref_x1, ref_y1 = ref_mask1.shape
        ref_image1 = zoom(ref_image1, (224 / ref_x1, 224 / ref_y1), order=3)
        ref_mask1 = zoom(ref_mask1, (224 / ref_x1, 224 / ref_y1), order=0)


        case_num, slice_num = mask_path.split('/')[-2:]
        # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        # ref_mask_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse',self.dataset+'_result',self.dataset, self.mode, 'Mask',case_num,slice_num)
        
        dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+self.mode+'.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(mask_path)
        #如果ref_mask_path不存在，则从json文件中读取
        # if not os.path.exists(ref_mask_path):
        ref_mask_path2 = dict[folder_name]
        ref_image_path2 = ref_mask_path2.replace('Mask', 'CT')


        ref_image2 = np.array(Image.open(ref_image_path2).convert("L"), dtype=np.float32)
        ref_mask2 = np.array(Image.open(ref_mask_path2).convert("L"), dtype=np.float32)
        ref_image2 = (ref_image2 - ref_image2.min()) / (ref_image2.max() - ref_image2.min())
        ref_mask2 = ref_mask2 / 255.0

        ref_x2, ref_y2 = ref_mask2.shape
        ref_image2 = zoom(ref_image2, (224 / ref_x2, 224 / ref_y2), order=3)
        ref_mask2 = zoom(ref_mask2, (224 / ref_x2, 224 / ref_y2), order=0)


        target_idx = int(os.path.basename(image_path).split('.')[0])
        random_idx = int(os.path.basename(ref_image_path1).split('.')[0]) 
        largest_idx = int(os.path.basename(ref_image_path2).split('.')[0])
        #计算距离绝对值
        distance1 = abs(target_idx - random_idx)
        distance2 = abs(target_idx - largest_idx)

        sample = {
            'image': image,
            'label': mask,
            'ref_image1': ref_image1,
            'ref_mask1': ref_mask1,
            'ref_image2': ref_image2,
            'ref_mask2': ref_mask2,
            'distance1': distance1,
            'distance2': distance2
        }

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = os.path.basename(os.path.dirname(image_path))
        sample['image_path'] = image_path

        return sample


class Dataset_v6_4(Dataset):  # 5 fixed pairs per slice, total length = 5 * num_slices / n(n-1); Except the largest slice, also try all the remaining slices(n(n-1))
    def __init__(self, base_dir, mode, transform=None, k=5):
        self.transform = transform
        self.data_dir = base_dir
        self.mode = mode
        self.k = k  # number of references per slice

        self.dataset = base_dir.split('/')[-2]
        if self.k > 0:
            self.pairs = self._build_k_pairs()
        else:
            self.pairs = self._build_all_pairs()

    def _build_k_pairs(self):
        """构建所有的 (query, reference) 对"""
        pairs = []
        ct_root = os.path.join(self.data_dir, "CT")
        mask_root = os.path.join(self.data_dir, "Mask")

        for patient_folder in os.listdir(ct_root):
            ct_folder = os.path.join(ct_root, patient_folder)
            mask_folder = os.path.join(mask_root, patient_folder)
            slice_files = sorted(os.listdir(ct_folder))
            slice_paths = [
                (os.path.join(ct_folder, f), os.path.join(mask_folder, f))
                for f in slice_files if os.path.exists(os.path.join(mask_folder, f))
            ]

            for query_path, query_mask in slice_paths:
                # 取其他 slice 中的 k 个作为 reference（不含自己）
                other_refs = [p for p in slice_paths if p[0] != query_path]
                k_refs = random.sample(other_refs, min(self.k, len(other_refs)))
                for ref_path, ref_mask in k_refs:
                    pairs.append((query_path, query_mask, ref_path, ref_mask))

        return pairs
    
    def _build_all_pairs(self):
        """构建所有的 (query, reference) 对，ref 为同 volume 中其他所有 slice"""
        pairs = []
        ct_root = os.path.join(self.data_dir, "CT")
        mask_root = os.path.join(self.data_dir, "Mask")

        for patient_folder in os.listdir(ct_root):
            ct_folder = os.path.join(ct_root, patient_folder)
            mask_folder = os.path.join(mask_root, patient_folder)
            slice_files = sorted(os.listdir(ct_folder))
            slice_paths = [
                (os.path.join(ct_folder, f), os.path.join(mask_folder, f))
                for f in slice_files if os.path.exists(os.path.join(mask_folder, f))
            ]

            for query_path, query_mask in slice_paths:
                for ref_path, ref_mask in slice_paths:
                    if query_path != ref_path:
                        query_idx = int(os.path.basename(query_path).split('.')[0])
                        ref_idx = int(os.path.basename(ref_path).split('.')[0])
                        distance = query_idx - ref_idx
                        pairs.append((query_path, query_mask, ref_path, ref_mask, distance))

        return pairs


    def __len__(self):
        return len(self.pairs)

    def _load_and_process(self, img_path, mask_path):
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0

        h, w = image.shape
        image = zoom(image, (224 / h, 224 / w), order=3)
        mask = zoom(mask, (224 / h, 224 / w), order=0)

        # image = torch.tensor(image).unsqueeze(0).float()
        # mask = torch.tensor(mask).unsqueeze(0).float()
        return image, mask

    def __getitem__(self, idx):
        query_img_path, query_mask_path, ref_img_path, ref_mask_path, distance = self.pairs[idx]

        query_img, query_mask = self._load_and_process(query_img_path, query_mask_path)
        ref_img, ref_mask = self._load_and_process(ref_img_path, ref_mask_path)

        case_num, slice_num = query_mask_path.split('/')[-2:]
        # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        # ref_mask_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse',self.dataset+'_result',self.dataset, self.mode, 'Mask',case_num,slice_num)
        
        dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+self.mode+'.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(query_mask_path)
        #如果ref_mask_path不存在，则从json文件中读取
        # if not os.path.exists(ref_mask_path):
        ref_mask_path2 = dict[folder_name]
        ref_image_path2 = ref_mask_path2.replace('Mask', 'CT')
        ref_img2, ref_mask2 = self._load_and_process(ref_image_path2, ref_mask_path2)

        target_idx = int(os.path.basename(query_img_path).split('.')[0])
        random_idx = int(os.path.basename(ref_img_path).split('.')[0]) 
        largest_idx = int(os.path.basename(ref_image_path2).split('.')[0])

        #计算距离绝对值
        distance1 = abs(target_idx - random_idx)
        distance2 = abs(target_idx - largest_idx)

        sample = {
            'image': query_img,
            'label': query_mask,
            'ref_image1': ref_img,
            'ref_mask1': ref_mask,
            'ref_image2': ref_img2,
            'ref_mask2': ref_mask2,
            'distance1': distance1,
            'distance2': distance2
            # 'case_name': os.path.basename(os.path.dirname(query_img_path)),
            # 'image_path': query_img_path,
            # 'ref_path': ref_img_path
        }

        
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = os.path.basename(os.path.dirname(query_img_path))
        sample['image_path'] = query_img_path
        

        return sample


# class Dataset_v6_5(Dataset):  # use adjacent slice as reference
#     def __init__(self, base_dir, mode, transform=None):
#         self.transform = transform
#         self.data_dir = base_dir
#         self.mode = mode
#         self.dataset = base_dir.split('/')[-2]
#         self.image_paths, self.mask_paths = self._get_image_mask_paths()
#         self.ref_dict = self._build_ref_dict()

#     def _get_image_mask_paths(self):
#         image_paths = []
#         mask_paths = []
#         ct_dir = os.path.join(self.data_dir, "CT")

#         for patient_folder in os.listdir(ct_dir):
#             patient_ct_folder = os.path.join(ct_dir, patient_folder)
#             for ct_filename in os.listdir(patient_ct_folder):
#                 ct_path = os.path.join(patient_ct_folder, ct_filename)
#                 mask_path = ct_path.replace('CT', 'Mask')
#                 if os.path.exists(mask_path):
#                     image_paths.append(ct_path)
#                     mask_paths.append(mask_path)
        
#         return image_paths, mask_paths

#     def _build_ref_dict(self):
#         ref_dict = {}
#         ct_dir = os.path.join(self.data_dir, "CT")
#         for patient_folder in os.listdir(ct_dir):
#             patient_ct_folder = os.path.join(ct_dir, patient_folder)
#             all_slices = sorted(os.listdir(patient_ct_folder))
#             num_slices = len(all_slices)

#             for i, filename in enumerate(all_slices):
#                 curr_path = os.path.join(patient_ct_folder, filename)

#                 # Skip if this image isn't in our list
#                 if curr_path not in self.image_paths:
#                     print('Skipping:', curr_path)
#                     continue

#                 # Choose adjacent slice as reference
#                 if i < num_slices // 2:
#                     ref_idx = min(i + 1, num_slices - 1)
#                 else:
#                     ref_idx = max(i - 1, 0)
#                 ref_path = os.path.join(patient_ct_folder, all_slices[ref_idx])
#                 ref_dict[curr_path] = ref_path
#         # print(ref_dict)
#         return ref_dict

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         mask_path = self.mask_paths[idx]

#         image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
#         mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
#         image = (image - image.min()) / (image.max() - image.min())
#         mask = mask / 255.0

#         x, y = image.shape
#         image = zoom(image, (224 / x, 224 / y), order=3)
#         mask = zoom(mask, (224 / x, 224 / y), order=0)

#         ref_image_path = self.ref_dict[image_path]
#         ref_mask_path = ref_image_path.replace('CT', 'Mask')

#         ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
#         ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
#         ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
#         ref_mask = ref_mask / 255.0

#         ref_x, ref_y = ref_mask.shape
#         ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
#         ref_mask = zoom(ref_mask, (224 / ref_x, 224 / ref_y), order=0)

#         sample = {
#             'image': image,
#             'label': mask,
#             'ref_image': ref_image,
#             'ref_mask': ref_mask
#         }

#         if self.transform:
#             sample = self.transform(sample)
#         sample['case_name'] = os.path.basename(os.path.dirname(image_path))
#         sample['image_path'] = image_path

#         return sample



class Dataset_v6_5(Dataset):   # use NEIGHBOR slice as reference image (from json)
    def __init__(self, base_dir, mode, transform=None):
        self.transform = transform
        self.data_dir = base_dir
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        self.dataset = base_dir.split('/')[-2]
        self.mode = mode

        self.ref_map = {}
        dict_path = os.path.join(self.data_dir, "annotation_dict_neighbor.json")
        if os.path.exists(dict_path):
            print(f"[INFO] neighbor-slice json found: {dict_path}")
            with open(dict_path, "r") as f:
                self.ref_map = json.load(f)
        else:
            raise FileNotFoundError(f"[FATAL] neighbor-slice json not found: {dict_path}")

    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, "CT")
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            if not os.path.isdir(patient_ct_folder):
                continue
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                if not os.path.isfile(ct_path):
                    continue
                mask_path = ct_path.replace('CT', 'Mask')
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
                else:
                    # 你认为全标注，所以缺一个就报错更符合“终端报错”的诉求
                    raise FileNotFoundError(
                        f"[FATAL] Missing mask for CT slice:\n  CT: {ct_path}\n  Mask: {mask_path}"
                    )
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path  = self.mask_paths[idx]

        # ---- 原始加载 & 归一化/缩放（保持不变）----
        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        mask  = mask / 255.0
        x, y = image.shape
        image = zoom(image, (224 / x, 224 / y), order=3)
        mask  = zoom(mask,  (224 / x, 224 / y), order=0)

        # ========== [CHANGED] 选取 reference：用 neighbor-slice JSON（按 slice 级别） ==========
        patient = os.path.basename(os.path.dirname(image_path))   # CT/<patient>/<file>
        slice_fname = os.path.basename(image_path)

        if patient not in self.ref_map:
            raise KeyError(f"[FATAL] Patient '{patient}' not found in neighbor json.")
        if slice_fname not in self.ref_map[patient]:
            raise KeyError(f"[FATAL] Slice '{slice_fname}' not found in neighbor json for patient '{patient}'.")

        entry = self.ref_map[patient][slice_fname]
        ref_image_path = entry.get("ref_ct_path", None)
        ref_mask_path  = entry.get("ref_mask_path", None)

        if not ref_image_path or not os.path.exists(ref_image_path):
            raise FileNotFoundError(
                f"[FATAL] ref_ct_path missing or not exists.\n"
                f"  patient={patient}, slice={slice_fname}\n"
                f"  ref_ct_path={ref_image_path}"
            )
        if not ref_mask_path or not os.path.exists(ref_mask_path):
            raise FileNotFoundError(
                f"[FATAL] ref_mask_path missing or not exists.\n"
                f"  patient={patient}, slice={slice_fname}\n"
                f"  ref_mask_path={ref_mask_path}"
            )

        # ---- 加载 reference 并做同样预处理（保持不变）----
        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        ref_mask  = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min() + 1e-8)
        ref_mask  = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        ref_mask  = zoom(ref_mask,  (224 / ref_x, 224 / ref_y), order=0)

        sample = {
            'image': image,
            'label': mask,
            'ref_image': ref_image,
            'ref_mask': ref_mask
        }
        if self.transform:
            sample = self.transform(sample)
        sample['case_name']  = patient
        sample['image_path'] = image_path
        sample['orig_size'] = np.array([x, y])  # original H, W before zoom to 224
        return sample
