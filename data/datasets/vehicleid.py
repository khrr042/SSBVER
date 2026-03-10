"""
import os
import os.path as osp

from .base import BaseImageDataset

class VehicleID(BaseImageDataset):
    def __init__(self, args, data_path, verbose=True):
        super(VehicleID).__init__()
        split_mapper = {'small':'800', 'medium':'1600', 'large': '2400'}
        split = split_mapper[args.split]
        self.data_path = data_path
        train_file = osp.join(data_path, 'train_list.txt')
        gallery_file = osp.join(data_path, 'Gallery_{}.txt'.format(split))
        query_file = osp.join(data_path, 'Query_{}.txt'.format(split))

        self.trainset = self._process_dir(train_file, relabel=True)
        self.galleryset = self._process_dir(gallery_file, relabel=False)
        self.queryset = self._process_dir(query_file, relabel=False)

        self.num_vids, self.num_imgs, self.num_cams = \
                                        self.get_imagedata_info(self.trainset)
        self.num_vids_q, self.num_imgs_q, self.num_cams_q = \
                                        self.get_imagedata_info(self.queryset)
        self.num_vids_g, self.num_imgs_g, self.num_cams_g = \
                                        self.get_imagedata_info(self.galleryset)
        
        if verbose:
            self.print_dataset_statistics(args, self.trainset, 
                                                self.queryset, 
                                                self.galleryset)

    
    def _process_dir(self, label_fl, relabel=False):
        items = open(label_fl, 'r').readlines()
        data = []
        vids = set()
        for item in items:
            vid = item.split(' ')[0].split('/')[-2]
            im_path = osp.join(self.data_path, item.split(' ')[0])
            data.append((im_path, int(vid), -1))
            vids.add(int(vid))
        
        vid2label = {vid: label for label, vid in enumerate(vids)}
        dataset = []
        for item in data:
            im_path, vid, camid = item
            if relabel:
                vid = vid2label[vid]
            dataset.append((im_path, vid, camid))
            
        return dataset

"""

import os
import os.path as osp
from collections import defaultdict

from .base import BaseImageDataset
from collections import defaultdict

class VehicleID(BaseImageDataset):
    def __init__(self, args, data_path, verbose=True):
        super(VehicleID, self).__init__()

        split_mapper = {
            'small': '800',
            'medium': '1600',
            'large': '2400'
        }
        split = split_mapper[args.split]
        self.data_path = data_path

        self.train_file = osp.join(data_path, 'train_test_split', 'train_list.txt')
        self.test_file = osp.join(data_path, 'train_test_split', f'test_list_{split}.txt')
        self.img2vid_file = osp.join(data_path, 'attribute', 'img2vid.txt')

        self.img2vid = self._read_img2vid(self.img2vid_file)

        self.trainset = self._process_train(self.train_file, relabel=True)
        self.queryset, self.galleryset = self._process_test(self.test_file)

        self.num_vids, self.num_imgs, self.num_cams = \
            self.get_imagedata_info(self.trainset)
        self.num_vids_q, self.num_imgs_q, self.num_cams_q = \
            self.get_imagedata_info(self.queryset)
        self.num_vids_g, self.num_imgs_g, self.num_cams_g = \
            self.get_imagedata_info(self.galleryset)

        if verbose:
            self.print_dataset_statistics(
                args, self.trainset, self.queryset, self.galleryset
            )

    def _read_img2vid(self, img2vid_path):
        img2vid = {}
        with open(img2vid_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_name, vid = line.split()
                img2vid[img_name] = int(vid)
        return img2vid

    def _read_image_list(self, list_file):
        img_list = []
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

            img_name = line.split()[0]
            img_name = osp.splitext(osp.basename(img_name))[0]
            img_list.append(img_name)

        return img_list

    def _process_train(self, list_file, relabel=False):
        data = []
        vids = set()

        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

            parts = line.split()
            img_key = osp.splitext(osp.basename(parts[0]))[0]

            if len(parts) > 1:
                vid = int(parts[1])
            else:
                vid = self.img2vid[img_key]

            im_path = osp.join(self.data_path, 'image', img_key + '.jpg')
            data.append((im_path, vid, -1))
            vids.add(vid)

        vid2label = {vid: label for label, vid in enumerate(sorted(vids))}

        dataset = []
        for im_path, vid, camid in data:
            if relabel:
                vid = vid2label[vid]
            dataset.append((im_path, vid, camid))

        return dataset


    def _process_test(self, list_file):
        vid2imgs = defaultdict(list)

        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                img_key = osp.splitext(osp.basename(parts[0]))[0]

                if len(parts) > 1:
                    vid = int(parts[1])
                else:
                    vid = self.img2vid[img_key]

                im_path = osp.join(self.data_path, 'image', img_key + '.jpg')
                vid2imgs[vid].append(im_path)

        queryset = []
        galleryset = []

        for vid, img_paths in vid2imgs.items():
            img_paths = sorted(img_paths)

            queryset.append((img_paths[0], vid, -1))
            for im_path in img_paths[1:]:
                galleryset.append((im_path, vid, -1))

        return queryset, galleryset