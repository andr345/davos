import os
import time
import torch

from collections import OrderedDict
from davos.sys.trainers import BaseTrainer
from davos.sys.stats import AverageMeter, StatValue
from davos.sys.logging import get_logger
from davos.lib import TensorDict
# noinspection PyUnresolvedReferences
from davos.lib.debug import DebugVis
#import torch.profiler

class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, checkpoint_interval=1):
        """
        :param actor:        Network training actor object
        :param loaders:      list of dataset loaders, e.g. [train_loader, val_loader].
                             In each epoch, the trainer runs one epoch for each loader.
        :param optimizer:    The optimizer used for training, e.g. Adam
        :param settings:     Training settings
        :param lr_scheduler: Learning rate scheduler
        :param validate_first:  HACK: Whether to run the second (val_loader) first.
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler, checkpoint_interval=checkpoint_interval)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})
        # Initialize tensorboard
        self.logger = get_logger('tensorboard', project_name=self.settings.project_path,
                                 experiment_name=self.settings.project_path, group_names=[l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        has_actor_loss_fn = hasattr(self.actor, "loss")
        has_actor_upload_fn = hasattr(self.actor, "upload")
        has_actor_iter_fn = hasattr(self.actor, "iterate")

        self._init_timing()

        do_profiling = False

        if do_profiling:
            print("Profiling is enabled.")

        # with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA] if do_profiling else [],
        #         schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler("/home/andreas/workspace/tensorboard/profiling"),
        #         profile_memory=False, with_stack=True, with_flops=True,
        #         record_shapes=True) as prof:
        if True:
            for i, data in enumerate(loader, 1):

                if isinstance(data, list):
                    # This is an uncollated batch, with one list entry per batch-sample, will require actor-custom uploader
                    data = dict(batch=data)

                data = TensorDict(data)
                # get inputs
                if self.move_data_to_gpu:
                    if has_actor_upload_fn:
                        data = self.actor.upload(data, self.device)
                    else:
                        data = data.to(self.device)

                data['epoch'] = self.epoch
                data['settings'] = self.settings

                if has_actor_iter_fn:
                    # The actor is responsible for .backward() and only returns stats
                    self.optimizer.zero_grad()
                    stats = self.actor.iterate(data, training=loader.training)
                    self.optimizer.step()
                else:
                    # forward pass
                    if has_actor_loss_fn:
                        # DataParallel will gather data from the forward,
                        # so that actor.loss() sees the full batch
                        loss, stats = self.actor.loss(data, self.actor(data))
                    else:
                        loss, stats = self.actor(data)

                    # backward pass and update weights
                    if loader.training:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                # prof.step()

                # update statistics
                batch_size = data['train_images'].shape[loader.stack_dim]
                self._update_stats(stats, batch_size, loader)

                # print statistics
                self._print_stats(i, loader, batch_size)

                if do_profiling:
                    if i == 20:
                        break

            # print()
            # if do_profiling:
            #     prof.export_chrome_trace("/home/andreas/pth_chrome_trace.dat")
            #     print()

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if (self.epoch % loader.epoch_interval == 0):
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._log_stats()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'sps: %.1f (%.1f),  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _log_stats(self):
        self.logger.log_training_epoch(self.stats, self.epoch)
