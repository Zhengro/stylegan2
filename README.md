## StyleGAN2

See official [README](https://github.com/NVlabs/stylegan2) for details.

## Requirements

All requirements are listed in [environment.yml](https://github.com/Zhengro/stylegan2/blob/master/environment.yml). Create the conda environment from the file:
```
conda env create -f environment.yml
```
Activate the new environment: 
```
conda activate stylegan2_env
```
StyleGAN2 relies on custom TensorFlow ops that are compiled on the fly using [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html). To test that your NVCC installation is working correctly, run:

```.bash
nvcc test_nvcc.cu -o test_nvcc -run
| CPU says hello.
| GPU says hello.
```

## Using pre-trained networks

All official pre-trained networks are stored as `*.pkl` files on the [StyleGAN2 Google Drive folder](https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7). 
Download [stylegan2-ffhq-config-f.pkl](http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl) (StyleGAN2 for [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) at 1024×1024) into the repo folder.

### Troubleshooting

1. If it throws the following error when running scripts in [Usage](https://github.com/Zhengro/stylegan2#usage):

   `tensorflow.python.framework.errors_impl.NotFoundError: /stylegan2/dnnlib/tflib/_cudacache/fused_bias_act_17c8f8342cd479bd10932d2c3c12c42c.so: undefined symbol:    _ZN10tensorflow12OpDefBuilder6OutputESs`

   Open [custom_ops.py](https://github.com/Zhengro/stylegan2/blob/master/dnnlib/tflib/custom_ops.py), find "--compiler-options \'-fPIC -D_GLIBCXX_USE_CXX11_ABI=0" and change it to "--compiler-options \'-fPIC -D_GLIBCXX_USE_CXX11_ABI=1"

2. It is a common problem that the first inference on Tensorflow is very slow, which means the time used to generate the first image is much longer than the sequential ones (See [Example of image generation](https://github.com/Zhengro/stylegan2#usage)). To handle this (when integrating stylegan2 into the GUI), we can deliberately generate a dummy image during the initialization phase (right after loading the model) as a warm-up.

### Usage

The results are placed in `results/`.

```
# Example of image generation
python run_generator.py generate-images --network=stylegan2-ffhq-config-f.pkl --seeds=0-9 --truncation-psi=1.0

# Example of style mixing
python run_generator.py style-mixing-example --network=stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0

# Example of gradual change
python run_generator.py style-mixing-gradual-change --network=stylegan2-ffhq-config-f.pkl --row-seeds=100 --col-seeds=85 --truncation-psi=1.0

# Example of noise effect
python run_generator.py style-mixing-noise --network=stylegan2-ffhq-config-f.pkl --row-seeds=100 --col-seeds=85 --col-styles=0-1 --randomize-noise True --truncation-psi=1.0

# Example of style mixing of three references
python run_generator.py style-mixing-multiple --network=stylegan2-ffhq-config-f.pkl --row-seeds=100 --col-seeds=85,20 --truncation-psi=1.0
```

## References

* [Analyzing and Improving the Image Quality of StyleGAN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf)
