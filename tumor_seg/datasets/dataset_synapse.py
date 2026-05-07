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
