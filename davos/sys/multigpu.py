from torch.nn.parallel import DistributedDataParallel, DataParallel

def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, DataParallel, DistributedDataParallel))


class MultiGPU(DataParallel):
    """Wraps a network to allow simple multi-GPU training."""
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)

class MultiGPU_ddp(DistributedDataParallel):
    """Wraps a network to allow simple multi-GPU training."""
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)
