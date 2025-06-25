# LightGlue ONNX

Open Neural Network Exchange (ONNX) compatible implementation of [LightGlue: Local Feature Matching at Light Speed](https://github.com/cvg/LightGlue). The ONNX model format allows for interoperability across different platforms with support for multiple execution providers, and removes Python-specific dependencies such as PyTorch. Supports TensorRT and OpenVINO.

> ✨ ***Original Repo***: This is an adapted version of the repo taken from [Fabio-Sim](https://github.com/fabio-sim), and for further information read this [blog post](https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/). 

## ⭐ Repo's Difference
This repo adapts the original code from [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) to work with Python 3.8.10 in a NVIDIA Orin NX with Jetpack 5.1.2 L4T 35.4.1, and ROS Noetic!

## ⭐ Instruction for installation
To install the repo it is sufficient run the command to create a conda environment using the environment.yaml file provided
```shell
conda env create -f jetson_env.yaml
conda activate lightglue_jetson
```
For Pytorch a specific version for the Jetson needs to be installed. In my case the version is **2.1.0a0+41361538.nv23.06**. 
Please be sure to check [NVIDIA's support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html). 
Once you have checked the supported version based on your hardware and your Jetpack version (my case 5.1.2), follow this link here:
```
https://developer.download.nvidia.cn/compute/redist/jp/
```
It is going to redirect you to a page with the different version of Jetpack. So click on your version (if my version is 5.1.2 --> click on v512 ), then pytorch and there should be your own cuda enabled pytorch version.
In order to install it in your environemt run the following command (in my case is this) with the env activated:
```shell
pip install --no-cache https://developer.download.nvidia.cn/compute/redist/jp/v512/https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl 
```
The original instructions can be found on NVIDIA's [webpage](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

Another tricky part of the installation is to find a compliant version of onnxruntime that has gpu. In this [link](https://elinux.org/Jetson_Zoo#ONNX_Runtime) can be found the pip wheel for your python and jetpack version. Dowload it and then run 

```shell
pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
```

The final step is to install a compliant version of TensorRT. In order to do so just follow the instructions provided by NVIDIA [here](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) and follow your preferred installation method.

# ⚠️⚠️⚠️ 
One thing that sometimes happens is that your TensorRT is not found when you create a virtual environment on the Jetson. To make sure that everything is found run the following command while your environemnt is activated.
```shell
export PYTHONPATH=/usr/lib/python3.8/dist-packages:$PYTHONPATH
```
Of course, the command depends on where your python distro is installed.

My TensorRT version = **8.5.2.2**

If there is anything missing in the steps please open and issues!

## ⭐ ONNX Export & Inference

For inference and export, I have both used the provided [typer](https://github.com/tiangolo/typer) CLI [`dynamo.py`](dynamo.py) and the existing weights found [here](https://github.com/fabio-sim/LightGlue-ONNX/releases).

```shell
$ python dynamo.py --help

Usage: dynamo.py [OPTIONS] COMMAND [ARGS]...

LightGlue Dynamo CLI

╭─ Commands ───────────────────────────────────────╮
│ export   Export LightGlue to ONNX.               │
│ infer    Run inference for LightGlue ONNX model. │
| trtexec  Run pure TensorRT inference using       |
|          Polygraphy.                             |
╰──────────────────────────────────────────────────╯
```

Pass `--help` to see the available options for each command. The CLI will export the full extractor-matcher pipeline so that you don't have to worry about orchestrating intermediate steps.

## 📖 Example Commands

<details>
<summary>🔥 ONNX Export</summary>
<pre>
python dynamo.py export superpoint \
  --num-keypoints 1024 \
  -b 2 -h 1024 -w 1024 \
  -o weights/superpoint_lightglue_pipeline.onnx
</pre>
</details>

<details>
<summary>⚡ ONNX Runtime Inference (CUDA)</summary>
<pre>
python dynamo.py infer \
  weights/superpoint_lightglue_pipeline.onnx \
  assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg \
  superpoint \
  -h 1024 -w 1024 \
  -d cuda
</pre>
</details>

<details>
<summary>🚀 ONNX Runtime Inference (TensorRT)</summary>
<pre>
python dynamo.py infer \
  weights/superpoint_lightglue_pipeline.trt.onnx \
  assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg \
  superpoint \
  -h 1024 -w 1024 \
  -d tensorrt --fp16
</pre>
</details>

<details>
<summary>🧩 TensorRT Inference</summary>
<pre>
python dynamo.py trtexec \
  weights/superpoint_lightglue_pipeline.trt.onnx \
  assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg \
  superpoint \
  -h 1024 -w 1024 \
  --fp16
</pre>
</details>

<details>
<summary>🟣 ONNX Runtime Inference (OpenVINO)</summary>
<pre>
python dynamo.py infer \
  weights/superpoint_lightglue_pipeline.onnx \
  assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg \
  superpoint \
  -h 512 -w 512 \
  -d openvino
</pre>
</details>

## Credits
If you use any ideas from the papers or code in this repo, please consider citing the authors of [LightGlue](https://arxiv.org/abs/2306.13643) and [SuperPoint](https://arxiv.org/abs/1712.07629) and [DISK](https://arxiv.org/abs/2006.13566). Lastly, if the ONNX versions helped you in any way, please also consider starring the [original repository](https://github.com/fabio-sim/LightGlue-ONNX) that helped me get started.

```txt
@inproceedings{lindenberger23lightglue,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  title     = {{LightGlue}: Local Feature Matching at Light Speed},
  booktitle = {ArXiv PrePrint},
  year      = {2023}
}
```

```txt
@article{DBLP:journals/corr/abs-1712-07629,
  author       = {Daniel DeTone and
                  Tomasz Malisiewicz and
                  Andrew Rabinovich},
  title        = {SuperPoint: Self-Supervised Interest Point Detection and Description},
  journal      = {CoRR},
  volume       = {abs/1712.07629},
  year         = {2017},
  url          = {http://arxiv.org/abs/1712.07629},
  eprinttype    = {arXiv},
  eprint       = {1712.07629},
  timestamp    = {Mon, 13 Aug 2018 16:47:29 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1712-07629.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```txt
@article{DBLP:journals/corr/abs-2006-13566,
  author       = {Michal J. Tyszkiewicz and
                  Pascal Fua and
                  Eduard Trulls},
  title        = {{DISK:} Learning local features with policy gradient},
  journal      = {CoRR},
  volume       = {abs/2006.13566},
  year         = {2020},
  url          = {https://arxiv.org/abs/2006.13566},
  eprinttype    = {arXiv},
  eprint       = {2006.13566},
  timestamp    = {Wed, 01 Jul 2020 15:21:23 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2006-13566.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
