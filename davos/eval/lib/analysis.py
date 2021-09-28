import io
from pathlib import Path
from dataclasses import dataclass
from zipfile import ZipFile
import json
from collections import OrderedDict as odict

import torch.multiprocessing as mp
import numpy as np
import pandas as pd

from davos.data.image_loader import imread_indexed
from davos.eval.lib.sequence import get_dataset, dataset_sequence_names
from davos.lib import measures


@dataclass
class EvaluationData:
    dset_name: str
    seq_name: str
    segmentations: dict  # segmentation label file paths
    annotations: dict  # ground-truth label file paths
    start_frames: dict  # dict: {object_id: first_frame_index}
    measure: str  # evaluation metric (J,F or a function)
    cached_scores: Path  # Path to cached scores, (might not exist)
    zipped_labels: Path  # Path to zipped labels, (might not exist)


def evaluate_sequence(eval_data: EvaluationData):
    """ Evaluate video sequence segmentations.  Originally db_eval_sequence() in the davis challenge toolkit. """

    segmentations = eval_data.segmentations
    annotations = eval_data.annotations
    start_frames = eval_data.start_frames
    measure = eval_data.measure
    cached_scores = eval_data.cached_scores
    zipped_labels = eval_data.zipped_labels

    results = dict(raw=odict(), dset_name=eval_data.dset_name, seq_name=eval_data.seq_name)

    _measure = measure if callable(measure) else \
        {'J': measures.davis_jaccard_measure, 'F': measures.davis_f_measure}[measure]
    _statistics = {'decay': measures.decay, 'mean': measures.mean, 'recall': measures.recall, 'std': measures.std}

    try:
        if cached_scores is not None and cached_scores.exists():
            # Try loading cached scores
            cached_scores = json.load(open(cached_scores))
            # Try finding the measure name in the scores of the first object in the first frame (ugly, I know)
            have_cached_scores = measure in next(iter(next(iter(cached_scores.values())).values())).keys()
        else:
            # No cached scores exist, load images from disk
            if zipped_labels is not None and zipped_labels.exists():
                # Read from zip file
                with ZipFile(zipped_labels, "r") as zip_file:
                    for name, file in segmentations.items():
                        file = file.relative_to(file.parents[1])  # sequence_name / frame_name.png
                        buf = zip_file.read(str(file))
                        segmentations[name] = imread_indexed(io.BytesIO(buf))
            else:
                # Read plain files
                for name, file in segmentations.items():
                    segmentations[name] = imread_indexed(file)

            for name, file in annotations.items():
                annotations[name] = imread_indexed(file)
            have_cached_scores = False

        for obj_id, first_frame in start_frames.items():

            r = np.ones((len(annotations))) * np.nan

            for i, (an, sg) in enumerate(zip(annotations, segmentations)):
                if list(annotations.keys()).index(first_frame) < i < len(annotations) - 1:
                    if have_cached_scores:
                        r[i] = cached_scores[an][str(obj_id)][measure]
                    else:
                        r[i] = _measure(segmentations[sg] == obj_id, annotations[an] == obj_id)

            results['raw'][obj_id] = r

        for stat, stat_fn in _statistics.items():
            results[stat] = [float(stat_fn(r)) for r in results['raw'].values()]
        results['status'] = 'ok'

    except Exception as e:
        # print(f"Evaluation of {seq_name} failed: {e}", file=sys.stderr)
        results['status'] = f'failure: {e}'

    return results


def evaluate_dataset(results_path, dset_name, measure='J', to_file=True, sequences=None, quiet=False):

    dset = get_dataset(dset_name, quiet=quiet)
    dset_scores = []
    dset_decay = []
    dset_recall = []

    if to_file:
        f = open(results_path / ("evaluation-%s.txt" % measure), "w")

    def _print(msg):
        if not quiet:
            print(msg)
        if to_file:
            print(msg, file=f)

    if sequences is not None:
        sequences = [sequences] if not isinstance(sequences, (list, tuple)) else sequences

    evaluations = []

    for j, sequence in enumerate(dset):
        if (sequences is not None) and (sequence.name not in sequences):
            continue

        seq_name = sequence.name
        frames = sequence.ground_truth_seg
        annotations = dict()
        segmentations = dict()
        start_frames = dict()

        # Find object ids and their start-frames
        for f_id, d in sequence.init_data.items():
            for obj_id in d['object_ids']:
                start_frames[int(obj_id)] = Path(d['mask']).stem
        if 0 in start_frames:  # Remove background
            start_frames.pop(0)

        # Get paths to files to load
        for f in frames:
            if f is None:
                continue
            file = Path(f)
            annotations[file.stem] = file
            if (results_path / dset_name).exists():  # New layout (separate directories per dataset)
                segmentations[file.stem] = results_path / dset_name / seq_name / file.name
            else:
                segmentations[file.stem] = results_path / seq_name / file.name

        cached_scores_path = results_path / dset_name / seq_name / "scores.json"
        zipped_labels_path = results_path / dset_name / seq_name / "labels.zip"

        evaluations.append(EvaluationData(dset_name, seq_name, segmentations, annotations, start_frames, measure, cached_scores_path, zipped_labels_path))

    # Start evaluations

    with mp.get_context('fork').Pool(16) as pool:
        results = pool.map(evaluate_sequence, evaluations)
        pool.close()
        pool.join()

    # Check for failures

    failed_sequences = list(sorted([r['seq_name'] for r in results if r['status'] != 'ok']))
    if len(failed_sequences) > 0:
        if to_file:
            f.close()
        return f"failed from sequence '{failed_sequences[0]}'"

    # Compute stats

    target_names = []
    for r in results:

        n_objs = len(r['raw'])
        seq_name = r['seq_name']
        _print("%d/%d: %s: %d object%s" % (j + 1, len(results), seq_name, n_objs, "s" if n_objs > 1 else ""))

        # Print scores, per frame and object, ignoring NaNs

        per_obj_score = []  # Per-object accuracies, averaged over the sequence
        per_frame_score = []  # Per-frame accuracies, averaged over the objects

        for obj_id, score in r['raw'].items():
            target_names.append('{}_{}'.format(seq_name, obj_id))
            per_frame_score.append(score)
            s = measures.mean(score)  # Sequence average for one object
            per_obj_score.append(s)
            if n_objs > 1:
                _print("joint {obj}: acc {score:.3f} ┊{apf}┊".format(obj=obj_id, score=s, apf=text_bargraph(score)))

        # Print mean object score per frame and final score
        dset_decay.extend(r['decay'])
        dset_recall.extend(r['recall'])
        dset_scores.extend(per_obj_score)

        seq_score = measures.mean(per_obj_score)  # Final score
        seq_mean_score = measures.nanmean(np.array(per_frame_score), axis=0)  # Mean object score per frame

        # Print sequence results
        _print("final  : acc {seq:.3f} ({dset:.3f}) ┊{apf}┊".format(
            seq=seq_score, dset=np.mean(dset_scores), apf=text_bargraph(seq_mean_score)))

    _print("%s: %.3f, recall: %.3f, decay: %.3f" % (measure, measures.mean(dset_scores), measures.mean(dset_recall), measures.mean(dset_decay)))

    if to_file:
        f.close()

    return target_names, dset_scores, dset_recall, dset_decay


def evaluate_vos(results_path, dataset='yt2019_jjval', quiet=False):

    global_results_file = results_path / f'{dataset}_global_results.csv'
    per_sequence_results_file = results_path / f'{dataset}_per-sequence_results.csv'

    r = evaluate_dataset(results_path, dataset, measure='J', to_file=False, quiet=quiet)
    if isinstance(r, str):
        raise RuntimeError(r)

    _, J, Jr, Jd = r
    seq_names, F, Fr, Fd = evaluate_dataset(results_path, dataset, measure='F', to_file=False, quiet=quiet)

    seq_res = [('J-Mean', J), ('J-Recall', Jr), ('J-Decay', Jd),
               ('F-Mean', F), ('F-Recall', Fr), ('F-Decay', Fd)]
    g_res = {k: measures.mean(v) for k, v in seq_res}
    g_res = {'G-Mean': (g_res['J-Mean'] + g_res['F-Mean']) * 0.5, **g_res}

    table_g = pd.DataFrame([g_res])
    with open(global_results_file, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
        # table_g.to_csv(sys.stdout, index=False, float_format="%.3f")

    seq_res = odict([('Sequence', seq_names), *seq_res])
    seq_res = [dict(zip(seq_res, t)) for t in zip(*seq_res.values())]  # Dict of lists -> list of dicts
    table_seq = pd.DataFrame(seq_res)
    with open(per_sequence_results_file, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")


def zip_dataset_sequences(dset_name, src_tree, dst_zip_file, quiet=False, dry_run=False, skip_ytvos_all_frames=False):
    """ Zip evaluation results, e.g. for uploading to the codalab server.
    :param dset_name:  Name of the dataset, eg yt2019_valid_all or dv2017_test_dev.
                       This is needed since the required zip-file directory structure varies by benchmark.
    :param src_tree:   Source directory containing the sequences to archive. Other files and directories inside, will be ignored.')
    :param dst_zip_file:   Destination zip filename path.
    :param skip_ytvos_all_frames:  Whether to only include every 5 frames, as needed on the codalab server. (YouTubeVOS only)
    """

    all_files = []

    anno_path = "Annotations/" if dset_name.startswith("yt") else ""

    if not quiet:
        print(f"Scanning for sequences belonging to {dset_name}")
    sequences = dataset_sequence_names(dset_name, quiet=quiet)

    if skip_ytvos_all_frames and dset_name.startswith("yt20") and dset_name.endswith("_all"):
        small_dset = get_dataset(dset_name[:-4])  # Dataset without the _all
        small_dset = {seq.name: set([Path(f).stem for f in seq.frames]) for seq in small_dset}
    else:
        small_dset = dict()

    for seq, n in sequences:

        zipped_labels = (src_tree / dset_name / seq / "labels.zip")
        if zipped_labels.exists():
            with ZipFile(zipped_labels, "r") as zip_file:
                frames = [Path(f) for f in zip_file.namelist() if f.endswith(".png") and f.startswith(seq)]
        else:
            zipped_labels = None
            if (src_tree / dset_name).exists():  # New layout
                frames = list(sorted((src_tree / dset_name / seq).glob("*.png")))
            else:
                frames = list(sorted((src_tree / seq).glob("*.png")))

        if len(frames) != n:
            s = f"missing output in sequence '{seq}'"
            if not dry_run:
                raise RuntimeError(s)
            else:
                print(s)

        if skip_ytvos_all_frames and seq in small_dset:
            # Remove frames not in the small dataset variant
            frames2 = []
            for f in frames:
                if f.stem in small_dset[seq]:
                    frames2.append(f)
            frames = frames2

        for src_file in frames:
            all_files.append((zipped_labels, src_file, f"{anno_path}{seq}/{src_file.name}"))

        raw_scores = (src_tree / dset_name / seq / "scores.json")
        if raw_scores.exists():
            all_files.append((None, raw_scores, f"{anno_path}{seq}/scores.json"))

    if dry_run:
        return

    for src_file in src_tree.glob(f"{dset_name}_*.csv"):
        dst_file = src_file.name
        all_files.append((None, src_file, dst_file))

    current_zip = None
    current_zip_name = None

    if not quiet:
        print(f"Creating {dst_zip_file}")

    with ZipFile(dst_zip_file, "w") as f:
        for src_ar, src, dst in all_files:
            if src_ar is None:
                f.write(src, arcname=dst)
            else:
                src_ar = str(src_ar)
                if current_zip is None or current_zip_name != src_ar:
                    current_zip = ZipFile(src_ar, "r")
                    current_zip_name = src_ar
                f.writestr(dst, current_zip.read(str(src)))


def text_bargraph(values):

    blocks = np.array(('u', ' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', 'o'))
    nsteps = len(blocks)-2-1
    hstep = 1 / (2*nsteps)
    values = np.array(values)
    nans = np.isnan(values)
    values[nans] = 0  # '░'
    indices = ((values + hstep) * nsteps + 1).astype(np.int)
    indices[values < 0] = 0
    indices[values > 1] = len(blocks)-1
    graph = blocks[indices]
    graph[nans] = '░'
    graph = str.join('', graph)
    return graph
