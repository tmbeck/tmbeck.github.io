---
aliases:
- /education/fastai/2020/09/05/the-story-so-far
author: <a href='https://www.linkedin.com/in/tmbeck'>Tim Beck</a>
badges: true
categories:
- education
- fastai
comments: false
date: '2020-09-05'
description: Review & notes from the first two lessons of fastai v4.
keywords: fastai, computing
layout: post
title: The story so far
toc: true

---

# The Story so Far

Ok, so this all started around July 2020 when I decided to invest in my own education and learn more about what some of my team members were working on. One of them suggested [fast.ai](https://fast.ai) so I decided to give it a try.

At the time, the course v3 was in "production". Since then, v4 has been released.

## Setup

Being the hands on engineer that I am, I decided to build my own system for machine learning from spare parts around the house. I already have experience deploying jupyter for use by development teams and I am no stranger to Linux. The fast.ai Getting Started guide originally suggested using managed services such as Google Cloud or AWS, but it was straightforward enough to get something working that I've documented it below.

Note that this isn't recommended by Jeremy in the video lesson, but I decided to do it to understand what is going on behind the scenes (and why pay AWS when I have the hardware and free solar energy).

### Hardware

These are parts I had laying around the house. The key if you want GPU acceleration with `pytorch` is to have the right GPU; too old and your GPU won't support [the necessary GPU compute capabilities](https://developer.nvidia.com/cuda-gpus). Below is the harware I had on hand:

* Intel i7-4770
* 32 GB of RAM
* NVIDIA Corporation GM204 [GeForce GTX 970] (rev a1); GPU compute capability 5.2
* 1 TB SSD (actually an upgrade; spinning rust was unbearably slow due to the low number of random IOPS)

### Software

Linux is my preferred operating system in general for hacking, so I went with the recently released Ubuntu 20.04 LTS Server. 

I've been using [anaconda](https://www.anaconda.com/products/individual) as my python distribution of choice for many years now, so I grabbed the python 3.8 x86_64 for Linux package.

I also grabbed [docker.io](https://docs.docker.com/engine/install/ubuntu/). Nvidia has a solution for doing GPU compute that requires docker to be installed, as well.

Since fastai uses pytorch, and pytorch only support CPU or GPU via cudatoolkit, I needed the `nvidia.ko` kernel module, the necessary nvidia cuda toolkit libraries, and the right packages in a conda environment.

I installed `nvidia-dkms-450` to provide nvidia drivers for my GTX 970.

[Nvidia provides a CUDA toolkit 10.2 Ubuntu repo](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork). Even though it says 18.04, it worked fine for me on 20.04. Just follow the instructions to install the `cuda` package.

You might also find the [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-linux/index.html) helpful for troubleshooting installation issues.

Finally, to take advantage of the GPU you must install the GPU-accelerated version of [pytorch](https://pytorch.org/get-started/locally/). Only the conda instructions are needed since the system python isn't used.

```bash
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Reboot as needed.

### Integration & Test

Finally we can create a new conda environment, install the necessary packages, and verify pytorch can see and use the GPU.

1. Create a fastai conda environment with python 3.8

    ```bash
    $ conda create -n fastai python=3.8
    $ conda activate fastai
    ```

2. Install fastai in conda, as per their [Install Instructions](https://docs.fast.ai/). It might take a while for `conda` to determine which channel to get the packages from.

    ```bash
    $ conda install -c fastai -c pytorch -c anaconda fastai gh anaconda
    $ conda install -c fastai -c pytorch -c anaconda fastai gh anaconda cudatoolkit=10.2
    Collecting package metadata (current_repodata.json): done
    Solving environment: done

    ## Package Plan ##

      environment location: /home/tbeck/anaconda3

      added / updated specs:
        - anaconda
        - cudatoolkit=10.2
        - fastai
        - gh


    The following packages will be downloaded:

        package                    |            build
        ---------------------------|-----------------
        ca-certificates-2020.7.22  |                0         132 KB  anaconda
        certifi-2020.6.20          |           py37_0         159 KB  anaconda
        conda-4.8.4                |           py37_0         3.0 MB  anaconda
        cudatoolkit-10.2.89        |       hfd86e86_1       540.0 MB  anaconda
        gh-0.11.1                  |                0         5.5 MB  fastai
        openssl-1.1.1g             |       h7b6447c_0         3.8 MB  anaconda
        ------------------------------------------------------------
                                              Total:       552.5 MB

    The following NEW packages will be INSTALLED:

      gh                 fastai/linux-64::gh-0.11.1-0

    The following packages will be SUPERSEDED by a higher-priority channel:

      ca-certificates                                 pkgs/main --> anaconda
      certifi                                         pkgs/main --> anaconda
      conda                                           pkgs/main --> anaconda
      cudatoolkit                                     pkgs/main --> anaconda
      openssl                                         pkgs/main --> anaconda


    Proceed ([y]/n)?
    ```

3. Once installed, you can quickly test that your GPU is seen and used from the command line like so:

    ```bash
    $ python -c 'import torch; print(torch.cuda.get_device_name())'
    GeForce GTX 970
    $ python -c 'import torch; print(torch.rand(2,3).cuda())'
    tensor([[0.3352, 0.0835, 0.5349],
            [0.3712, 0.2851, 0.8767]], device='cuda:0')
    ```

If you see your expected video card and a `tensor` returned, you're all set. If you have multiple GPU's installed, you may need to specify which one to use. Check out [this stack overflow article](https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter) on how to set the appropriate environment variables for the command line and for jupyter to work.

### A note on updates

I've noticed that if you update conda using `conda update --all`, it will try to pull in the latest version of `cudatoolkit`, which as of this writing is `cudatoolkit-11.0.221-h6bb024c_0`. This is safe to do, but you will need to downgrade back to `cudatoolkit-10.2`. This seems to be due to how anaconda handles/prioritizes packages from various channels. Below is an example.

#### Upgrading conda (only showing cudatoolkit for visbility - your output will differ)

```bash
$ conda update --all
...
The following packages will be UPDATED:

  cudatoolkit        anaconda::cudatoolkit-10.2.89-hfd86e8~ --> pkgs/main::cudatoolkit-11.0.221-h6bb024c_0
```

#### Downgrading cudatoolkit

```bash
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
...
  added / updated specs:
    - cudatoolkit=10.2

The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    cudatoolkit-10.2.89        |       hfd86e86_1       365.1 MB
    ------------------------------------------------------------
                                           Total:       365.1 MB

The following packages will be DOWNGRADED:

  cudatoolkit                           11.0.221-h6bb024c_0 --> 10.2.89-hfd86e86_1
```

#### Troubleshooting

If you get the below trying to use `torch` then cuda isn't working as expected.

```bash
$ python -c 'import torch; print(torch.rand(2,3).cuda())'
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/tbeck/anaconda3/envs/fastai/lib/python3.7/site-packages/torch/cuda/__init__.py", line 192, in _lazy_init
    _check_driver()
  File "/home/tbeck/anaconda3/envs/fastai/lib/python3.7/site-packages/torch/cuda/__init__.py", line 95, in _check_driver
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
```

Note that trying to install `cudatoolkit=10.2` alone might result in this error, so be sure that `pytorch` and `torchvision` are included when specifying `cudatoolkit=10.2`.

### Preparing for class

There are two sets of notebooks for the class:

* The `fastbook`, a guided set of notebooks with prose for following along in the videos: [fastbook](https://github.com/fastai/fastbook)
* The same notebooks as a study aid: [course-v4](https://github.com/fastai/course-v4)

Please consider showing your support by buying the fastbook: [Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527)

I setup my own area for hacking:

```bash
$ mkdir ~/src
$ cd ~/src && git clone https://github.com/fastai/fastbook.git && git clone https://github.com/fastai/course-v4.git
```

Now that CUDA is working and we have the code, I prefer fire up my `jupyter notebook` in a `screen` session. To do this I generated a [https://jupyter-notebook.readthedocs.io/en/stable/config.html](default jupyter notebook config) via `jupyte notebook --generate-config` and wrote it to `~/.jupyter/jupyte_notebook_config.py`. Then I made it listen on `0.0.0.0` so I can reach it from my LAN (or anywhere in the world via wireguard!).

Now I just run `screen`, activate conda with `conda activate fastai`, and finally start jupyter with `jupyter notebook`. For tricks on using screen see [this quickreference](https://gist.github.com/jctosta/af918e1618682638aa82)

If you prefer to use jupyter lab, you'll need to `conda install jupyterlab` and run `jupyter lab` instead.
