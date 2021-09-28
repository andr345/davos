import io
import json
import os
import importlib
from pathlib import Path
from time import time

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile

from torch.utils.data import DataLoader, Dataset

from davos.lib.measures import davis_jaccard_measure, davis_f_measure
from davos.eval.lib.sequence import Sequence, get_dataset
from .common import Frame
from davos.config import env_settings
from davos.data.image_loader import imwrite_indexed
from davos.sys.stats import AverageMeter
from davos.lib.debug import DebugVis

import torch.multiprocessing as mp
from time import sleep
import queue


def postprocess_frames(iq: mp.Queue, oq: mp.Queue, seq_name, zip_filename=None,
                       compute_scores=False, scores_filename=None, save_segs=False):
    # Background task for post-processing frames

    seq_scores = []

    zip_file = None
    if zip_filename is not None:
        zip_file = ZipFile(zip_filename, "w")

    while True:
        try:
            frame = iq.get(timeout=0.1)
        except queue.Empty:
            continue

        if frame is None:
            break

        object_ids = frame['object_ids']
        frame_name = frame['frame_name']
        labels = frame['labels']
        gt_labels = frame['gt_labels']
        raw_segs = frame['raw_segs']
        merged_segs = frame['merged_seg_probs']

        scores = dict()
        if compute_scores:
            for obj_id in object_ids:
                fg_mask = (labels == obj_id).squeeze()
                gt_mask = (gt_labels == obj_id).numpy()
                J = davis_jaccard_measure(fg_mask, gt_mask)
                F = davis_f_measure(fg_mask, gt_mask)
                scores[obj_id] = dict(J=J, F=F)

            if scores_filename is not None:
                seq_scores.append((frame_name, scores))

        labels_png = io.BytesIO()
        imwrite_indexed(labels_png, labels)
        labels_png.seek(0)
        labels_png = labels_png.read()

        segs_pth = None
        if save_segs and len(raw_segs) > 0:
            raw_segs = {obj_id: seg.half() for obj_id, seg in raw_segs.items()}
            merged_segs = {obj_id: seg.half() for obj_id, seg in merged_segs.items()}
            segs_pth = io.BytesIO()
            torch.save(dict(raw=raw_segs, merged=merged_segs), segs_pth)
            segs_pth.seek(0)
            segs_pth = segs_pth.read()

        r = dict(frame_name=frame_name, scores=scores)
        if zip_file is None:
            r['labels_png'] = labels_png
            if segs_pth is not None:
                r['segs_pth'] = segs_pth
        else:
            zip_file.writestr(f"{seq_name}/{frame_name}.png", labels_png)
            if segs_pth is not None:
                zip_file.writestr(f"{seq_name}/segs_{frame_name}.pth", segs_pth)

        oq.put(r)

    if zip_file is not None:
        zip_file.close()

    if compute_scores and scores_filename is not None:
        seq_scores = {s[0]: s[1] for s in seq_scores}
        json.dump(seq_scores, open(scores_filename, "w"))


class LTRTracker:

    def __init__(self, name, params_fn):

        env = env_settings()

        self.name = name
        self.segmentation_dir = env.segmentation_path / self.name
        self.raw_segs_dir = env.segmentation_path / "scores" / self.name
        self.params = params_fn()

        tracker_module = importlib.import_module(f'davos.eval.trackers.{self.params.tracker}')
        self.create_tracker = tracker_module.get_tracker_class()

        self.net = self.params.net
        self.device = 'cuda' if self.params.use_gpu else 'cpu'
        self.static_target = self.create_tracker(0, self.params)
        self.have_track2 = hasattr(self.static_target, 'track2')

        self.targets = dict()
        self.prev_frame = Frame()

    def track(self, image, labels=None, new_targets=None, frame_name=None, reset=False):

        if new_targets is None:
            new_targets = dict()
        if reset:
            self.targets = dict()
            self.prev_frame = Frame()

        frame = Frame(image, labels, self.device)

        # Add new target trackers
        for obj_id, data in new_targets.items():
            self.targets[obj_id] = self.create_tracker(obj_id, self.params)
            self.targets[obj_id].initialize(frame, self.prev_frame, bbox=new_targets[obj_id])

        # Update existing target trackers
        for obj_id, target in self.targets.items():
            if obj_id not in new_targets:
                target.track(frame, self.prev_frame)

        # Finalize if multi-target tracking
        if self.have_track2:
            for obj_id, target in self.targets.items():
                if obj_id not in new_targets:
                    target.track2(frame)

        # Optionally keep the raw segmentations, (logits) for debugging
        raw_segs = dict()
        if self.params.get('save_raw_segs', False):
            for obj_id, raw_seg in frame.segmentation_raw.items():
                raw_segs[obj_id] = raw_seg.clone()

        # Merge target predictions (raw scores will now be probabilities, not logits)
        frame = self.static_target.merge(frame, self.targets)
        self.prev_frame = frame

        merged_seg_probs = {k: v.clone() for k, v in frame.segmentation_raw.items()}
        r = dict(frame_name=frame_name, labels=frame.labels, bboxes=frame.target_bbox, raw_segs=raw_segs, merged_seg_probs=merged_seg_probs)
        return r

    def run_dataset(self, dataset_name, sequences=None, restart=None, skip_completed=False,
                    rank=0, world_size=1, compute_scores=False, save_to_zip=False, disable_progress_bar=False):
        """
        :param dataset_name:
        :param sequences:         Name (str) or names List[str] of sequences to evaluate. None means all
        :param restart:           Name of sequence to restart with
        :param skip_completed:    Whether to skip sequences that have been evaluated already
        :param verify_completed:  If skip_completed is True, ensure that all the files of sequences that would be skipped, can be read.
        :param rank:              The id of this multiprocessing instance. Used to select sequences when evaluating sequencess in parallel.
        :param world_size:        Number of parallel evaluations, when multiprocessing.
        :return:
        """

        have_progress_bar = ('SLURM_JOB_ID' not in os.environ) and not disable_progress_bar

        dataset = get_dataset(dataset_name)

        # Select sequences

        if sequences is not None:
            sequences = [sequences] if not isinstance(sequences, (list, tuple)) else sequences
            dset_sequences_by_name = {seq.name: dataset[i] for i, seq in enumerate(dataset)}
            dataset = [dset_sequences_by_name[seq] for seq in sequences]

        # Find sequence to restart from

        if restart is not None:
            seq_names = {seq.name for seq in dataset}
            if restart not in seq_names:
                print("Did not find sequence '%s', cannot restart" % restart)
                quit(0)
            print("Restarting from", restart)

        save_segs = self.params.get('save_raw_segs', False)

        for k, seq in enumerate(dataset):

            if restart is not None:
                if restart == seq.name:
                    restart = None
                else:
                    continue

            # Determine whether this tracker instance should process the sequence
            if k % world_size != rank:
                continue

            segmentation_dir = self.segmentation_dir / dataset_name / seq.name
            segmentation_dir.mkdir(exist_ok=True, parents=True)

            if skip_completed and (segmentation_dir / "done").exists():
                print(f"{k + 1}/{len(dataset)} skipping {seq.name} - completed")
                (segmentation_dir / "done").touch(exist_ok=True)
                continue

            desc = f"{k + 1}/{len(dataset)} {seq.name}"
            object_ids = list(map(int, seq.object_ids))

            torch.cuda.empty_cache()
            loader = DataLoader(SequenceLoader(dataset_name, seq, load_gt_seg=compute_scores), num_workers=2, pin_memory=True, collate_fn=SequenceLoader.collate_fn)

            if have_progress_bar:
                # Show a progress bar if not running on a cluster and not disabled by the user
                loader = tqdm(loader, desc=desc, total=len(seq.frames), unit='f')
            else:
                print(desc)

            seconds_per_frame = AverageMeter()

            # Signal the post-processor to immediately save the files
            zip_filename = segmentation_dir / "labels.zip"
            scores_filename = segmentation_dir / "scores.json"

            pp_input = mp.Queue()
            pp_output = mp.Queue()
            pp_worker = mp.Process(target=postprocess_frames, args=(pp_input, pp_output, seq.name),
                                   kwargs=dict(zip_filename=zip_filename, compute_scores=compute_scores,
                                               scores_filename=scores_filename, save_segs=save_segs))
            pp_worker.start()

            # Loop over the sequence

            runtime = dict()
            t0_load = time()
            vis = DebugVis(start_visdom=False)

            for i, (kwargs, gt_labels) in enumerate(loader):

                if i == 0 and have_progress_bar:
                    loader.start_t = loader._time()  # Discount the initial pause in the DataLoader

                try:
                    t0 = time()
                    runtime['load'] = round(t0 - t0_load, 1)

                    out = self.track(**kwargs, reset=(i == 0))

                    t1 = time()
                    runtime['frame'] = round(t1 - t0, 1)
                    seconds_per_frame.update(t1 - t0)

                except RuntimeError as e:
                    pp_input.put(None)
                    pp_worker.join()
                    pp_worker.close()
                    if e.args[0].startswith("CUDA out of memory"):
                        raise MemoryError(*e.args)
                    else:
                        raise e

                out['time'] = seconds_per_frame.val
                out['gt_labels'] = gt_labels
                out['object_ids'] = object_ids
                out['compute_scores'] = compute_scores

                # Submit for post processing (encoding labels as png, compute scores etc)

                t0 = time()

                while pp_input.qsize() > 2:
                    # print("Waiting for post processing")
                    sleep(0.5)
                pp_input.put(out)

                t1 = time()
                runtime['pp'] = round(t1 - t0, 1)

                # print(runtime)

                t0_load = t1

            pp_input.put(None)  # Force worker to quit when done

            if not have_progress_bar:
                print(f"{k + 1}/{len(dataset)} {seq.name}: fps={1.0 / seconds_per_frame.avg:.2}")

            pp_results = []
            while pp_output.qsize() > 0:
                pp_results.append(pp_output.get())

            pp_worker.join()
            (segmentation_dir / "done").touch(exist_ok=True)


class SequenceLoader(Dataset):

    def __init__(self, dataset_name, sequence: Sequence, load_gt_seg=False):
        self.dataset_name = dataset_name
        self.sequence = sequence
        self.load_gt_seg = load_gt_seg

    def __len__(self):
        return len(self.sequence.frames)

    def __getitem__(self, item):

        fname = self.sequence.frames[item]
        image = Image.open(fname)
        image = torch.tensor(np.atleast_3d(np.array(image)).transpose(2, 0, 1))
        frame_info = self.sequence.frame_info(item)

        gt_labels = None
        if self.load_gt_seg:
            if len(self.sequence.ground_truth_seg) != len(self.sequence.frames):
                raise RuntimeError(f"Cannot load ground labels of {self.sequence.dataset}, {self.sequence.name} ")
            gt_labels = torch.tensor(np.array(Image.open(self.sequence.ground_truth_seg[item])))

        new_targets = frame_info.get('init_object_ids', None)
        if new_targets is not None:
            new_targets = {int(obj_id): frame_info['init_bbox'][obj_id] for obj_id in new_targets}

        labels = frame_info.get('init_mask', None)
        if labels is not None:
            labels = torch.tensor(labels.copy())

        kwargs = dict(frame_name=Path(fname).stem, image=image, new_targets=new_targets, labels=labels)  # kwargs for the track() function
        return kwargs, gt_labels

    @staticmethod
    def collate_fn(samples):
        assert len(samples) == 1  # Batch size 1
        return samples[0]
