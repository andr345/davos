from pathlib import Path
import pkg_resources
from collections import OrderedDict, defaultdict
import json

import torch
import numpy as np
from PIL import Image

from .base_video_dataset import BaseVideoDataset
from davos.data.image_loader import imread_indexed
from davos.data.bounding_box_utils import masks_to_bboxes


class VOSMeta:
    def __init__(self, data=None, filename=None, quiet=False):

        self.quiet = quiet

        if filename is not None:
            self.load(filename)
        elif data is not None:
            self._data = data
        else:
            raise ValueError("Must set either data or filename parameter")

        # Ensure sequence names and frame numbers are strings and not numbers (which would break the data collation function)


    def save(self, gen_meta: Path):
        gen_meta.parent.mkdir(exist_ok=True, parents=True)
        json.dump(self._data, open(gen_meta, "w"))

    def load(self, gen_meta: Path):
        if not gen_meta.exists():
            print("Generated metadata file %s is not found." % gen_meta)
            print("Find and run VOSMeta.generate() to create it.")
            raise FileNotFoundError(gen_meta)
        self._data = json.load(open(gen_meta), object_pairs_hook=OrderedDict)

    @classmethod
    def generate(cls, dset_name: str, dset_images_path: Path, dset_annos_path: Path, deeper=False):
        """
        Count the annotation mask pixels per object, per frame, in all sequences in a dataset
        :param dset_name:        Dataset name, for printing the progress bar.
        :param dset_annos_path:  Path to annotations directory, containing sequence directories,
                                 with annotation frames in them.
        :param deeper:           If true: search deeper into the directory tree to find the frames.
        :return: Dataset meta dict:

        {'sequence0':
            {
             'shape': (height, width)

             'obj_sizes':  # Object pixels per frame
                {'frame0': {'object0': px_count, 'object1': px_count, ...},
                 'frame1': {'object0': px_count, 'object1': px_count, ...},
                ... },

             'bboxes':  # Bounding boxes per frame
                {'frame0': {'object0': bbox, 'object1': bbox, ...},
                 'frame1': {'object0': bbox, 'object1': bbox, ...},
                ... },
            ...
        }
        """
        assert(dset_annos_path.exists())

        if deeper:
            # Search recursively for directories with png-files, and derive the sequence names from there
            frames = list(sorted(dset_annos_path.rglob("*.png")))
            sequences = [str(s) for s in list(sorted(set([f.relative_to(dset_annos_path).parent for f in frames])))]
        else:
            sequences = [p.stem for p in sorted(dset_annos_path.glob("*")) if p.is_dir()]

        try:
            from tqdm import tqdm
        except:
            def tqdm(x, *args, **kwargs):
                return x

        dset_meta = OrderedDict()
        for seq in tqdm(sequences, desc=dset_name, unit="seq"):

            obj_sizes2 = defaultdict(OrderedDict)
            bboxes = defaultdict(OrderedDict)
            shape = None
            frame_names = [file.stem for file in sorted((dset_images_path / seq).glob("*.jpg"))]
            anno_paths = list(sorted((dset_annos_path / seq).glob("*.png")))

            # Extract information from the given label frames
            for path in anno_paths:
                f_id = path.stem

                # Count label-pixels per frame
                labels = imread_indexed(path)
                # labels = np.array(Image.open(path))
                obj_ids, obj_sizes = np.unique(labels, return_counts=True)
                obj_ids = [str(oid) for oid in obj_ids]
                obj_sizes = obj_sizes.tolist()

                if '0' in obj_ids:  # Remove background id
                    obj_ids = obj_ids[1:]
                    obj_sizes = obj_sizes[1:]
                obj_sizes2[f_id] = OrderedDict(zip(obj_ids, obj_sizes))

                # Generate per-label bounding boxes
                for obj_id in obj_ids:
                    bboxes[f_id][obj_id] = cls._mask_to_bbox(labels == int(obj_id))

                if shape is None:
                    shape = labels.shape[:2]

            # Format result
            dset_meta[seq] = dict(shape=shape, obj_sizes=obj_sizes2, bboxes=bboxes, frame_names=frame_names)

        return VOSMeta(dset_meta)


    @staticmethod
    def _mask_to_bbox(mask: np.ndarray):

        mask = mask.astype(int)
        xs = mask.sum(axis=-2).nonzero()[0].tolist()
        ys = mask.sum(axis=-1).nonzero()[0].tolist()

        if len(ys) > 0 and len(xs) > 0:
            x, y, w, h = xs[0], ys[0], xs[-1] - xs[0], ys[-1] - ys[0]
        else:
            x, y, w, h = 0, 0, 0, 0

        return [x, y, w, h]

    @staticmethod
    def _transpose_nested_dict(d):
        """ Permute a 2-level nested dict such that the inner and outer keys swap places. """
        d2 = defaultdict(OrderedDict)
        for key1, inner in d.items():
            for key2, value in inner.items():
                d2[key2][key1] = value
        return d2

    def select_split(self, dataset_name, split):

        sequences = pkg_resources.resource_string("davos", f"dataset/specs/{dataset_name}_{split}.txt").decode('utf-8')
        sequences = set([s.strip() for s in sequences.splitlines()])
        all_sequences = set(self._data.keys())
        to_remove = all_sequences.difference(sequences)
        for seq_name in to_remove:
            self._data.pop(seq_name)

    def get_sequence_names(self):
        return list(self._data.keys())

    def get_shape(self, seq_name):
        """ Sequence image shape (h,w) """
        h, w = self._data[seq_name]['shape']
        return h, w

    def get_obj_ids(self, seq_name):
        """ All objects in the sequence """
        return list(self.get_obj_sizes_per_object(seq_name).keys())

    def get_frame_names(self, seq_name):
        """ All filename stems of the frames in the sequence """
        return self._data[seq_name]['frame_names']

    def enable_all_frames(self, dset_images_path):
        """ For YouTubeVOS: Update the frame names with (jpeg) files from the <split>_all_frames set
        :param dset_images_path:  /path/to/train_all_frames/JPEGImages (or valid or test)
        :param seq: Sequence name
        :return:
        """

        # Try load the cached index
        idx_file = dset_images_path.parent / "frame_names.json"
        if idx_file.exists():
            if not self.quiet:
                print('Loading cached frame names from %s' % idx_file)
            all_frame_names = json.load(open(idx_file))
        else:
            # Cache the data to the user's home directory (guaranteed to be writable)
            all_frame_names = dict()
            user_idx_file = Path.home() / (dset_images_path.parent.stem + "_frame_names.json")
            print('Indexing YouTubeVOS "all_frames" frame names to %s' % user_idx_file)
            for seq in self._data:
                all_frame_names[seq] = [file.stem for file in sorted((dset_images_path / seq).glob("*.jpg"))]
            json.dump(all_frame_names, open(user_idx_file, "w"))
            print('Done. Move %s to %s to load faster next time.' % (user_idx_file, idx_file))

        for seq, frame_names in all_frame_names.items():
            self._data[seq]['frame_names'] = frame_names

    def get_aspect_ratio(self, seq_name):
        """ Sequence aspect ratio """
        h, w = self._data[seq_name]['shape']
        return w / h

    def get_obj_sizes_per_frame(self, seq_name):
        """ Get object pixel counts, grouped by frame names """
        return self._data[seq_name]['obj_sizes']

    def get_bboxes_per_frame(self, seq_name):
        """ Object bounding boxes, grouped by frame names """
        return self._data[seq_name]['bboxes']

    def get_obj_sizes_per_object(self, seq_name):
        """ Object pixel counts, grouped by object """
        return self._transpose_nested_dict(self.get_obj_sizes_per_frame(seq_name))

    def get_bboxes_per_object(self, seq_name):
        """ Object bounding boxes, grouped by object """
        return self._transpose_nested_dict(self.get_bboxes_per_frame(seq_name))

    @staticmethod
    def generate_datasets_meta(src, dst=Path("~/vosdataset_meta").expanduser()):
        VOSMeta.generate("SyntheticCoco", src / "JPEGImages", src / "Annotations").save(src / "generated_meta.json")


class VOSDatasetBase(BaseVideoDataset):

    """ Generic VOS dataset reader base class, for both DAVIS and YouTubeVOS """

    def __init__(self, name: str, root: Path, version=None, split='train',
                 multiobj=True, vis_threshold=10, with_all_labels=False, with_distractors=False):
        """
        :param root:            Dataset root path, eg /path/to/DAVIS or /path/to/YouTubeVOS/
                                Note: YouTubeVOS 2018 and 2019 are expected to be in
                                /path/to/YouTubeVOS/2018 and /path/to/YouTubeVOS/2019, respectively
        :param name:            'DAVIS' or 'YouTubeVOS' (case sensitive)
        :param version:         DAVIS: '2016', '2017, YouTubeVOS: '2018' or '2019'
        :param split:           DAVIS: Any name in DAVIS/ImageSets/<year>,
                                YouTubeVOS: 'test', 'train', 'valid' or 'jjtrain', 'jjvalid'
        :param multiobj:        Whether the dataset will return all objects in a sequence or
                                multiple sequences with one object in each.
        :param vis_threshold:   Minimum number of pixels required to consider a target object "visible".
        """

        if not (root.exists() and root.is_dir()):
            print(f"Cannot find dataset root '{root}'")
            raise RuntimeError

        super().__init__(name, root)

        self.version = version
        self.split = split
        self.vis_threshold = vis_threshold
        self.multiobj = multiobj
        self.with_distractors = with_distractors
        self.with_all_labels = with_all_labels

    def _load_image(self, path):
        im = np.atleast_3d(np.array(Image.open(path)))
        assert im is not None
        im = np.atleast_3d(im)
        return im

    @staticmethod
    def _load_anno(path):
        if not path.exists():
            return None
        # im = np.atleast_3d(np.array(Image.open(path)))
        im = imread_indexed(path)
        return im

    def get_num_sequences(self):
        return len(self._samples)

    def get_sequence_info(self, sample_id):
        """ Get sample meta data.
        :param sample_id:  Sample to query.
        :return: dict of metadata:
                sequence:    Sequence name
                frame_shape: (height, width) of the images
                frame_names: List of frame filename stems in the sequence
                object_ids:  Id numbers of all objects occurring in the sequence
                obj_sizes:   Matrix shape=(frames, object) of the number of pixels for each object in each frame
                             Coordinates in this matrix relate to the frame_names and object_ids
                visible:     Boolean matrix of the same shape as obj_sizes. Entries with more pixels
                             than self.visible_threshold are True.
        """
        m = self.gmeta
        seq_name, obj_ids = self._samples[sample_id]
        f_names = m.get_frame_names(seq_name)  # All frames

        f2i = {f: i for i, f in enumerate(f_names)}  # Frame name to matrix index
        o2i = {o: i for i, o in enumerate(obj_ids)}  # Object id to matrix index

        # Get a matrix of object sizes: shape=(frames, objects)
        obj_sizes = torch.zeros((len(f_names), len(obj_ids)), dtype=torch.int)
        sizes_per_object = m.get_obj_sizes_per_object(seq_name)

        for obj_id in obj_ids:
            frames = sizes_per_object[obj_id]
            oid = o2i[obj_id]
            for f, sz in frames.items():
                obj_sizes[f2i[f], oid] = sz

        visible = (obj_sizes > self.vis_threshold).byte()

        return dict(sequence=seq_name, frame_shape=m.get_shape(seq_name), frame_names=f_names, object_ids=obj_ids,
                    object_sizes=obj_sizes, visible=visible, valid=visible)

    def get_paths_and_bboxes(self, sequence_info):

        seq_name = sequence_info['sequence']
        annos_root = self._anno_path / seq_name
        images_root = self._jpeg_path / seq_name

        frame_names = sequence_info['frame_names']
        f2i = {f: i for i, f in enumerate(frame_names)}

        images = [str(images_root / (f + ".jpg")) for f in frame_names]

        # Find the frames where ground truth is available and
        # get the bounding boxes and segmentation labels of those frames
        all_bboxes = self.gmeta.get_bboxes_per_frame(seq_name)
        gt_labels = [str(annos_root / (f + ".png")) if f in all_bboxes.keys() else None for f in frame_names]

        gt_bboxes = OrderedDict()
        for obj_id in sequence_info['object_ids']:
            gt_bboxes[obj_id] = np.array([all_bboxes.get(frame, {}).get(obj_id, [-1, -1, -1, -1]) for frame in frame_names])

        return images, gt_labels, gt_bboxes

    def _construct_sequence(self, sequence_info):
        raise NotImplementedError

    def get_sequence_list(self):
        if len(self.sequence_list) > 0:
            return self.sequence_list
        self.sequence_list = [self._construct_sequence(self.get_sequence_info(i)) for i in range(len(self._samples))]
        return self.sequence_list

    def __len__(self):
        return len(self._samples)

    def _get_image_path(self, meta, frame_id):
        return self._jpeg_path / meta['sequence'] / (meta['frame_names'][frame_id] + ".jpg")

    def _get_anno_path(self, meta, frame_id):
        return self._anno_path / meta['sequence'] / (meta['frame_names'][frame_id] + ".png")

    def get_frames(self, sample_id, frame_ids, anno=None):
        """  Fetch frames with the given ids.
        :param sample_id:  Sample to get.
        :param frame_ids:  List of frame indices in the sequence belonging to the sample_id
        :return: dict of metadata and data:
                sequence:  Sequence name
                images:    List of images. No entries may be None
                labels:    List of label/mask images. Entries may be None if the data is missing
                bboxes:    List of bounding boxes. Entries may be None if the data is missing
        """
        seq_name, obj_ids = self._samples[sample_id]

        meta = self.get_sequence_info(sample_id) if anno is None else anno
        frame_names = meta['frame_names']
        if not self.is_synthetic_video_dataset():
            images = [self._load_image(self._jpeg_path / seq_name / (frame_names[f] + ".jpg")) for f in frame_ids]
            labels = [self._load_anno(self._anno_path / seq_name / (frame_names[f] + ".png")).copy() for f in frame_ids]
        else:
            images, labels = self._load_synthetic_frames(seq_name, [frame_names[f] for f in frame_ids])

        # Generate bounding boxes for the requested objects
        bboxes = []
        for lb in labels:
            lb = torch.from_numpy(lb.squeeze())
            frame_bbs = {}
            for obj_id in obj_ids:
                bbox = masks_to_bboxes(lb == int(obj_id), fmt='t')
                if bbox[3] == 0 or bbox[2] == 0:
                    print("!")
                frame_bbs[obj_id] = bbox
            bboxes.append(frame_bbs)

        # Insert empty bboxes for missing object ids
        for bbox in bboxes:
            for obj_id in obj_ids:
                if obj_id not in bbox:
                    bbox[obj_id] = torch.zeros(4, dtype=torch.float32)

        # Remap to object id 1, if requested - for training

        distractors = []
        all_labels = []

        if not self.multiobj:
            assert len(obj_ids) == 1
            obj_id = obj_ids[0]

            if self.with_all_labels:
                all_labels = [torch.from_numpy(lb) for lb in labels]

            labels2 = []
            for lb in labels:
                fg = (lb == int(obj_id))  # Training target
                if self.with_distractors:
                    bg = (lb == 0)  # Background without distractors
                    distractors.append(torch.Tensor(~(bg | fg)))
                labels2.append(torch.Tensor(fg))
            labels = labels2

            bboxes = [bbox[obj_id] for bbox in bboxes]
        else:
            all_labels = [torch.Tensor(lb) for lb in labels]

        object_meta = {key: meta[key] for key in ['sequence', 'frame_shape', 'frame_names', 'object_ids']}

        anno_frames = dict(bbox=bboxes, mask=labels)
        if 'main_obj_id' in meta:
            object_meta['main_obj_id'] = meta['main_obj_id']
        if len(all_labels) > 0:
            anno_frames['labels'] = all_labels
        if len(distractors) > 0:
            anno_frames['distractor_masks'] = distractors
        for key in ['object_sizes', 'visible', 'valid']:
            value = meta[key]
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return images, anno_frames, object_meta

    def get_name(self):
        return "%s/%s/%s" % (self.name, self.version, self.split)

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def get_num_classes(self):
        return 0

    def get_class_list(self):
        return []

    def get_sequences_in_class(self, class_name):
        raise []

    def has_segmentation_info(self):
        return True
