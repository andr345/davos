from pathlib import Path


def env_settings():
    return EnvSettings()


class EnvSettings:

    def __init__(self):
        datasets = Path.home() / "datasets"
        ws = Path.home() / "workspace"

        self.workspace_dir = ws

        self.tensorboard_dir = ws / "tensorboard"
        self.neptune_api_token = Path.home() / ".ssh/neptune_ai_api_token"
        self.neptune_user_id = Path.home() / ".ssh/neptune_user_id"

        self.checkpoints_dir = ws / 'checkpoints'
        self.network_path = ws / "davos_weights"
        self.ramdisk_path = Path("/dev/shm")

        results = ws / "results"
        self.all_results_path = results
        self.segmentation_path = results / "segmentation"

        self.davis_dir = datasets / 'DAVIS'
        self.davis_testdev_dir = datasets / 'DAVIS-full'
        self.youtubevos_dir = datasets / "YouTubeVOS"
        self.coco_dir = datasets / "coco"


def set_common_environment(name, vcuda=None, mp_spawn=True, deterministic=None):

    assert deterministic is None  # Not implemented yet

    if mp_spawn:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

    import os
    import torch
    import cv2

    run_name = name.split(".")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)  # This is needed to avoid strange crashes related to opencv
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match nvidia-smi and CUDA device ids
    if vcuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = vcuda

    os.environ['LTR_RUN_NAME'] = name

    if os.environ.get('TMUX_PANE') is not None:
        tmux_name = "-".join(run_name)
        os.system(f'tmux rename-session {tmux_name}')
        os.system(f'tmux rename-window ""')
        os.system(f"tmux set-option status-left-length {min(len(tmux_name)+3, 40)}")


def run_training(run_file, world_size=None, **kwargs):

    import argparse
    import davos.sys.settings as ws_settings
    import importlib

    name = Path(run_file).stem

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=name)
    parser.add_argument('--vcuda', type=str, default=kwargs.get('vcuda'), help='Visible cuda devices, e.g 0,1')
    args = parser.parse_args()
    set_common_environment("train." + name, vcuda=args.vcuda)

    print(f'Training {name}')

    settings = ws_settings.Settings()
    settings.script_name = name
    settings.project_path = f'davos/{name}'

    expr_module = importlib.import_module(f'davos.train.{name}')
    expr_func = getattr(expr_module, 'run')

    if world_size is None or world_size == 1:
        expr_func(settings)
    else:
        import torch.multiprocessing as mp
        mp.spawn(_run_training_ddp, args=(world_size, expr_func, settings), nprocs=world_size)

    return args

def _run_training_ddp(rank, world_size, main_function, settings):

    import os
    import torch.distributed as dist

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    main_function(rank, world_size, settings)

    dist.destroy_process_group()

def prepare_evaluation(run_file=None, **kwargs):

    import argparse
    run_name = Path(run_file).stem

    parser = argparse.ArgumentParser()
    parser.add_argument('--vcuda', type=str, default=kwargs.get('vcuda'), help='Visible cuda devices, e.g 0,1')
    parser.add_argument('--rank', type=int, default=kwargs.get('rank', 0), help='Parallel run: index of this run, in group of runs')
    parser.add_argument('--wsize', type=int, default=kwargs.get('wsize', 1), help='Parallel run: world size / group size')
    parser.add_argument('--restart', type=str, default=None, help='Sequence to restart from')
    parser.add_argument('--crash_restarts', type=int, default=0, help='Number of times the application should try to restart itself on a crash')
    args = parser.parse_args()

    set_common_environment(run_name, vcuda=args.vcuda)

    return run_name, args
