---
license: mit
language:
- en
tags:
- person-reid
- deep-learning
- image-recognition
---

This repo contains the pre-trained weights of OSNet, specialized for person recognition (i.e., person re-identification).

Related work:
- [Omni-Scale Feature Learning for Person Re-Identification](https://arxiv.org/abs/1905.00953)
- [Learning Generalisable Omni-Scale Representations for Person Re-Identification](https://arxiv.org/abs/1910.06827)

# Get started

Install the [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) package.

```bash
# cd to your preferred directory and clone this repo
git clone https://github.com/KaiyangZhou/deep-person-reid.git

# create environment
cd deep-person-reid/
conda create --name torchreid python=3.7
conda activate torchreid

# install dependencies
# make sure `which python` and `which pip` point to the correct path
pip install -r requirements.txt

# install torch and torchvision (select the proper cuda version to suit your machine)
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# install torchreid (don't need to re-build it if you modify the source code)
python setup.py develop
```

Use Torchreid as a feature extractor in your project.

```python
from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='a/b/c/model.pth.tar',
    device='cuda'
)

image_list = [
    'a/b/c/image001.jpg',
    'a/b/c/image002.jpg',
    'a/b/c/image003.jpg',
    'a/b/c/image004.jpg',
    'a/b/c/image005.jpg'
]

features = extractor(image_list)
print(features.shape) # output (5, 512)
```