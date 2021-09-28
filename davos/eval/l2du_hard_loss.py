import sys, pathlib
project_root = [str(p.parent) for p in pathlib.Path(__file__).resolve().parents if p.name == 'davos'][0]
sys.path.append(project_root)

from pathlib import Path
from davos.eval.trackers.ltr_tracker import LTRTracker
from davos.eval.lib.params import TrackerParams
from davos.sys.net_wrappers import load_net_with_backbone
from davos.config import env_settings


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.seg_to_bb_mode = 'var'
    params.max_scale_change = (0.95, 1.1)
    params.min_mask_area = 100

    params.use_gpu = True

    params.image_sample_size = (30 * 16, 52 * 16)
    params.search_area_scale = 5.0
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = None

    # Learning parameters
    params.sample_memory_size = 32
    params.learning_rate = 0.1
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 1

    # Net optimization params
    params.update_target_model = True
    params.net_opt_iter = 20
    params.net_opt_update_iter = 5
    params.update_distractor_fn = 'update_distractors_wta'

    net_path = Path.home() / 'workspace/checkpoints/ltr/lwl/l2du_hard_dloss_stage2/LWTLNet_ep0006.pth.tar'
    params.net = load_net_with_backbone(net_path=net_path, image_format='bgr255',
                                        mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0])

    params.tracker = 'dlwl'

    return params


def main():

    from davos.config import prepare_evaluation
    run_name, args = prepare_evaluation(__file__)
    tracker = LTRTracker(run_name, parameters)

    for dset_name in ('dv2017_val',):  # ('dv2017_val', 'dv2017_test_dev', 'yt2018_valid_all', 'yt2019_valid_all'):
        tracker.run_dataset(dataset_name=dset_name, rank=args.rank, world_size=args.wsize, skip_completed=True)


if __name__ == '__main__':
    main()
