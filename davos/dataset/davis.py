from pathlib import Path
from davos.dataset.vos_base import VOSDatasetBase, VOSMeta
from davos.eval.lib.sequence import Sequence
from davos.config import env_settings


class DAVIS(VOSDatasetBase):
    """ The Davis VOS dataset

        Publication:
            A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
            F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung
            CVPR, 2016
            http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf

        Download the dataset from https://davischallenge.org/davis2017/code.html
        """
    def __init__(self, root=None, sequences=None, version='2017', split='train', multiobj=True,
                 vis_threshold=10, with_all_labels=False, with_distractors=False,
                 quality='480p', quiet=False):
        """
        args:
             root - Dataset root path. If unset, it uses the path in your local.py config.
             sequences - List of sequence names. Limit to a subset of sequences if not None.
             version - '2016' or '2017
             split - Any name in DAVIS/ImageSets/<year>
             multiobj - Whether the dataset will return all objects in a sequence or multiple sequences with one object
                        in each.
             vis_threshold - Minimum number of pixels required to consider a target object "visible".
        """
        if version == '2017':
            if split in ['train', 'val']:
                root = env_settings().davis_dir if root is None else root
            elif split in ['test-dev']:
                root = env_settings().davis_testdev_dir if root is None else root
            else:
                raise Exception('Unknown split {}'.format(split))
        else:
            root = env_settings().davis16_dir if root is None else root
            
        super().__init__(name='DAVIS', root=Path(root), version=version, split=split, multiobj=multiobj,
                         vis_threshold=vis_threshold, with_all_labels=with_all_labels, with_distractors=with_distractors)

        if "potato" in quality:
            resolution = {'480potato': '480p', '720potato': '720p'}[quality]
        else:
            resolution = quality

        dset_path = self.root
        self._jpeg_path = dset_path / 'JPEGImages' / quality
        self._anno_path = dset_path / 'Annotations' / resolution
        self.quality = quality

        meta_path = dset_path / "generated_meta.json"
        if meta_path.exists():
            self.gmeta = VOSMeta(filename=meta_path)
        else:
            self.gmeta = VOSMeta.generate('DAVIS', self._jpeg_path, self._anno_path)
            self.gmeta.save(meta_path)

        if sequences is None:
            if self.split != 'all':
                fname = dset_path / 'ImageSets' / self.version / (self.split + '.txt')
                sequences = open(fname).read().splitlines()
            else:
                sequences = [p for p in sorted(self._jpeg_path.glob("*")) if p.is_dir()]

        self.sequence_names = sequences
        self._samples = []

        for seq in sequences:
            obj_ids = self.gmeta.get_obj_ids(seq)
            if self.multiobj:  # Multiple objects per sample
                self._samples.append((seq, obj_ids))
            else:  # One object per sample
                self._samples.extend([(seq, [obj_id]) for obj_id in obj_ids])

        if not quiet:
            print("%s loaded." % self.get_name())

    def _construct_sequence(self, sequence_info):

        seq_name = sequence_info['sequence']
        images, gt_labels, gt_bboxes = self.get_paths_and_bboxes(sequence_info)

        return Sequence(name=seq_name, frames=images, dataset='DAVIS', ground_truth_rect=gt_bboxes,
                        ground_truth_seg=gt_labels, object_ids=sequence_info['object_ids'],
                        multiobj_mode=self.multiobj)
