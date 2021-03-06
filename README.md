### Introduction

This is the code repository for the paper Distractor-Aware Video Object Segmentation.
This is a quick and only slightly cleaned-up version, so expect to do some manual
tweaking of paths and data to get it running.   

All paths, including those to the datasets are defined in `config.py`.

By default, checkpoints and experiment results will go in
`~/workspace/davos_weights/`, `~/workspace/checkpoints/` and `~/workspace/results/`
by default.

Network weights, including the ResNet50-backbone used as a starting point,
are available [here](https://liuonline-my.sharepoint.com/:f:/g/personal/andro44_liu_se/Eih6HFplqupBlpljmWMWZeYBieSZwHZakgwGiieKF2A1YA?e=Gdjcxe)
and should be unpacked into `~/workspace/davos_weights`. The path davos_weights/ exists in the zip-file.

To train: run any of the programs under davos/train (except actors.py).

To evaluate: run any of the programs under davos/eval. Which datasets to evaluate on are configured in the respective files. 

###

Please cite this paper if you reuse or refer to this work in an academic setting:
```
@inproceedings{davos2021,
    author={Robinson, Andreas and Eldesokey, Abdelrahman and Felsberg Michael},
    title={Distractor-Aware Video Object Segmentation},
    booktitle = {DAGM German Conference on Pattern Recognition, DAGM GCPR 2021},
    year={2021},
    publisher={Springer}
}
```

### Authors

Andreas Robinson and Abdelrahman Eldesokey

This code is based on the LWL method and uses part of the PyTracking framework (https://github.com/visionml/pytracking),
written by Martin Danelljan, Goutam Bhat, Christoph Mayer and Felix Järemo-Lawin.



