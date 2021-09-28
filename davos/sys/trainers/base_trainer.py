from pathlib import Path
import torch
from davos.sys import multigpu
from davos.sys.loading import default_pickler


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, checkpoint_interval=1):
        """
        :param actor:        Network training actor object
        :param loaders:      list of dataset loaders, e.g. [train_loader, val_loader].
                             In each epoch, the trainer runs one epoch for each loader.
        :param optimizer:    The optimizer used for training, e.g. Adam
        :param settings:     Training settings
        :param lr_scheduler: Learning rate scheduler
        :param checkpoint_interval: Number of epochs between saved checkpoints.
                             Note that if > 1, there will be a "latest" checkpoint file saved every epoch.
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders
        self.checkpoint_interval = checkpoint_interval

        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = Path(self.settings.env.workspace_dir).expanduser()
            self._checkpoint_dir = self.settings.env.checkpoints_dir
            self._checkpoint_dir.mkdir(exist_ok=True, parents=True)
            self.training_output_path = self.settings.env.checkpoints_dir / self.settings.project_path
            self.training_output_path.mkdir(exist_ok=True, parents=True)
        else:
            self._checkpoint_dir = None
            self.training_output_path = None

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)

    def train(self, max_epochs, load_latest=False):
        """ Train for the given number of epochs.
        :param max_epochs: Max number of training epochs,
        :param load_latest: Bool indicating whether to resume from latest epoch.
        """
        epoch = -1

        if load_latest:
            self.load_checkpoint()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.train_epoch()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            save_as_latest = ((epoch % self.checkpoint_interval != 0) or (epoch == 0)) and (epoch != max_epochs)
            self.save_checkpoint(as_latest=save_as_latest)

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def _atomic_save(self, state, fname):
        """ Save a checkpoint with a tmp suffix before renaming it to the real filename.
         This ensures the write is atomic, i.e either the whole checkpoint is is written, or nothing. """
        file = fname.with_suffix(".tmp")
        torch.save(state, file)
        file.rename(fname)

    def save_checkpoint(self, as_latest=False):
        """Save a checkpoint of the network and other variables."""

        if self.training_output_path is None:
            return

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net': net.state_dict(),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }

        latest_fname = self.training_output_path / f'{net_type}_latest.pth'
        fname = self.training_output_path / f'{net_type}_ep{self.epoch:04d}.pth'

        if as_latest:
            self._atomic_save(state, latest_fname)
        else:
            self._atomic_save(state, fname)
            if latest_fname.exists():
                latest_fname.unlink()

    def _find_latest_checkpoint(self, net_name, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self._checkpoint_dir / self.settings.project_path
        checkpoint_file = ckpt_path / (net_name + "_latest.pth")
        if not checkpoint_file.exists():
            files = list(sorted(ckpt_path.glob(net_name + "_ep*.pth")))
            files = list(sorted(ckpt_path.glob(net_name + "_ep*.pth")))
            if files:
                checkpoint_file = files[-1]
            else:
                return None
        return checkpoint_file

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False):
        """ Load a network checkpoint file """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_file = self._find_latest_checkpoint(net_type)
            if checkpoint_file is None:
                print('No matching checkpoint file found')
                return False
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_file = f'{self._checkpoint_dir}/{self.settings.project_path}/{net_type}_ep{checkpoint:04d}.pth'
        elif isinstance(checkpoint, (str, Path)):
            checkpoint = Path(checkpoint).expanduser()
            if checkpoint.is_dir():  # checkpoint is the path
                checkpoint_file = self._find_latest_checkpoint(net_type, checkpoint)
                if checkpoint_file is None:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_file = checkpoint
        else:
            raise TypeError

        # Load network
        checkpoint_dict = torch.load(str(checkpoint_file), map_location='cpu', pickle_module=default_pickler)
        #assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch

        return True
