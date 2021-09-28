import os
from pathlib import Path

from davos.config import env_settings
from davos.sys.stats import AverageMeter, StatValue


class BaseLogger:

    def log_training_epoch(self, stats: dict, epoch: int):

        for group, data in stats.items():  # e.g group='train' or group='val'
            if data is None:
                continue

            for key, val in data.items():

                if isinstance(val, AverageMeter):
                    if val.has_new_data:
                        val = val.history[-1]
                    elif val.count > 0:
                        val = val.avg
                    else:
                        continue

                elif isinstance(val, StatValue):
                    val = val.val

                if isinstance(val, str):
                    self.log_text(group=group, key=key, value=val, step=epoch)
                else:
                    self.log_scalar(group=group, key=key, value=val, step=epoch)

            self.flush(group)

    def log_evaluation(self, benchmark_name: str, results: dict, run_id: int = None):
        """
        :param benchmark_name:   Benchark name, eg dv2017_val
        :type benchmark_name: str
        :param results:  Dict of metrics, and more extensive results like text files or strings
                        e.g dict(J=0.8, F=0.7, G=0.75, Jdecay = ...,
                                 details=Path("/path/to/text_file"),
                                 other='hello world')
                        Values that are Path objects or strings are logged as text
        :type results: dict
        :param run_id:  Optional ID number, useful when running the same evaluation several times.
        :type run_id: int
        """
        for key, val in results.items():
            if isinstance(val, (Path, str)):
                self.log_text(benchmark_name, key, str(val), step=run_id)
            else:
                self.log_scalar(benchmark_name, key, val, step=run_id)
        self.flush(benchmark_name)

        raise NotImplementedError

    def log_scalar(self, group, key, value, step=None):
        raise NotImplementedError

    def log_text(self, group, key, value, step=None):
        raise NotImplementedError

    def flush(self, group):
        raise NotImplementedError


class MultiDestinationLogger(BaseLogger):

    def __init__(self, loggers):
        super().__init__()
        self.loggers = loggers

    def log_training_epoch(self, stats: dict, epoch: int):
        for logger in self.loggers:
            logger.log_epoch(stats, epoch)

    def log_evaluation(self, benchmark_name, results, run_id=None):
        for logger in self.loggers:
            logger.log_evaluation(benchmark_name, results, run_id=run_id)

    def log_scalar(self, group, key, value, step=None):
        for logger in self.loggers:
            logger.log_scalar(group, key, value, step)

    def log_text(self, group, key, value, step=None):
        for logger in self.loggers:
            logger.log_scalar(group, key, value, step)

    def flush(self, group):
        for logger in self.loggers:
            logger.flush(group)


class TensorboardLogger(BaseLogger):

    def __init__(self, project_name, experiment_name, group_names):
        super().__init__()

        from torch.utils.tensorboard import SummaryWriter
        self.log_path = env_settings().tensorboard_dir / project_name / experiment_name
        self.writers = {name: SummaryWriter(self.log_path / name) for name in group_names}

    def log_scalar(self, group, key, value, step=None):
        self.writers[group].add_scalar(key, value, step)

    def log_text(self, group, key, value, step=None):
        if isinstance(value, Path):
            value = "\n".join(open(value).readlines())
        self.writers[group].add_text(key, value, step)

    def flush(self, group):
        self.writers[group].flush()


class NeptuneLogger(BaseLogger):

    def __init__(self, project_name, experiment_name, group_names):
        super().__init__()

        import neptune
        self.neptune = neptune

        env = env_settings()

        if env.neptune_api_token.exists():
            api_token = open(env.neptune_api_token).readline().strip()
        elif hasattr(os.environ, 'NEPTUNE_API_TOKEN'):
            api_token = os.environ['NEPTUNE_API_TOKEN']
        else:
            raise ValueError

        if env.neptune_api_token.exists():
            user_id = open(env.neptune_user_id).readline().strip()
        elif hasattr(os.environ, 'NEPTUNE_API_TOKEN'):
            user_id = os.environ['NEPTUNE_USER_ID']  # Not an official Neptune.ai environment variable
        else:
            raise ValueError

        self.neptune.init(f"{user_id}/{project_name}", api_token)

        self.data_group_names = group_names
        self.experiment = self.neptune.create_experiment(experiment_name)

    def log_scalar(self, group, key, value, step=None):
        self.experiment.log_metric(f"{group}/{key}", x=step, y=value)

    def log_text(self, group, key, value, step=None):
        if isinstance(value, Path):
            value = "\n".join(open(value).readlines())
        if step is None:
            self.experiment.log_text(f"{group}/{key}", x=value)
        else:
            self.experiment.log_text(f"{group}/{key}", x=step, y=value)

    def flush(self, group):
        pass


def get_logger(backend='tensorboard', **kwargs):

    if isinstance(backend, str):
        backend = [backend]

    loggers = []

    if 'tensorboard' in backend:
        loggers.append(TensorboardLogger(**kwargs))

    if 'neptune' in backend:
        loggers.append(NeptuneLogger(**kwargs))

    if len(loggers) == 0:
        raise ValueError
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return MultiDestinationLogger(loggers)


def unit_test_training_logger():

    import numpy as np

    logger = get_logger('neptune', project_name='TEST-PROJECT',
                        experiment_name='train-exp1', group_names=['lr', ])

    for i in range(1000):
        stats = {"loss/seg": np.random.random(1),
                 "loss/total": np.random.random(1),
                 "acc": np.random.random(1),
                 "some_text": str(np.random.random(1))}
        logger.log_training_epoch(dict(train=stats), i)


if __name__ == '__main__':
    unit_test_training_logger()
