<div align="right"> <a href="https://github.com/fabio-sim/LightGlue-ONNX/blob/main/README.md">English</a> | <a href="https://github.com/fabio-sim/LightGlue-ONNX/blob/main/docs/README.zh.md">简体中文</a> | 日本語</div> 

[![ONNX](https://img.shields.io/badge/ONNX-grey)](https://onnx.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-76B900)](https://developer.nvidia.com/tensorrt)
[![GitHub Repo stars](https://img.shields.io/github/stars/fabio-sim/LightGlue-ONNX)](https://github.com/fabio-sim/LightGlue-ONNX/stargazers)
[![GitHub all releases](https://img.shields.io/github/downloads/fabio-sim/LightGlue-ONNX/total)](https://github.com/fabio-sim/LightGlue-ONNX/releases)
[![Blog](https://img.shields.io/badge/Blog-blue)](https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/)

# LightGlue ONNX

[LightGlue: Local Feature Matching at Light Speed](https://github.com/cvg/LightGlue) の ONNX（Open Neural Network Exchange）互換実装です。ONNX モデルフォーマットにより、複数の実行プロバイダーに対応し、さまざまなプラットフォーム間での相互運用性が向上します。また、PyTorch などの Python 固有の依存関係を排除します。TensorRT および OpenVINO をサポートしています。

> ✨ ***新機能***: エンドツーエンドの並列動的バッチサイズのサポート。詳細はこの [ブログ記事](https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/) をご覧ください。

<p align="center"><a href="https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/"><img src="../assets/inference-comparison-speedup.svg" alt="レイテンシ比較" width=90%></a><br><em>⏱️ 推論時間の比較</em></p>

<p align="center"><a href="https://arxiv.org/abs/2306.13643"><img src="../assets/easy_hard.jpg" alt="LightGlue 図" width=80%></a></p>

<details>
<summary>更新履歴</summary>

- **2024年7月17日**: エンドツーエンドの並列動的バッチサイズのサポート。スクリプト UX の改良。 [ブログ記事](https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/) を追加。
- **2023年11月2日**: 約30%のスピードアップのために ArgMax を最適化する TopK トリックを導入。
- **2023年10月4日**: FlashAttention-2 をサポートする `onnxruntime>=1.16.0` を使用した LightGlue ONNX モデルの統合。長いシーケンス長（キーポイントの数）で最大80%の推論速度向上。
- **2023年10月27日**: LightGlue-ONNX が [Kornia](https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.OnnxLightGlue) に追加されました。
- **2023年7月19日**: TensorRT のサポートを追加。
- **2023年7月13日**: Flash Attention のサポートを追加。
- **2023年7月11日**: Mixed Precision のサポートを追加。
- **2023年7月4日**: 推論時間の比較を追加。
- **2023年7月1日**: `max_num_keypoints` をサポートするエクストラクタを追加。
- **2023年6月30日**: DISK エクストラクタのサポートを追加。
- **2023年6月28日**: エンドツーエンドの SuperPoint+LightGlue エクスポート & 推論パイプラインを追加。
</details>

## ⭐ ONNX エクスポート & 推論

LightGlue を簡単に ONNX へエクスポートし、ONNX Runtime で推論を行うための [typer](https://github.com/tiangolo/typer) CLI [`dynamo.py`](/dynamo.py) を提供しています。すぐに推論を試したい場合は、[こちら](https://github.com/fabio-sim/LightGlue-ONNX/releases) からすでにエクスポートされた ONNX モデルをダウンロードできます。

```shell
$ python dynamo.py --help

Usage: dynamo.py [OPTIONS] COMMAND [ARGS]...

LightGlue Dynamo CLI

╭─ コマンド ───────────────────────────────────────╮
│ export   LightGlue を ONNX にエクスポートします。  │
│ infer    LightGlue ONNX モデルの推論を実行します。 │
| trtexec  Polygraphy を使用して純粋な TensorRT     |
|          推論を実行します。                        |
╰──────────────────────────────────────────────────╯
```

各コマンドのオプションを確認するには、`--help` を使用してください。CLI は完全なエクストラクタ-マッチャー パイプラインをエクスポートするため、中間ステップの調整に悩む必要はありません。

## 📖 使用例コマンド

<details>
<summary>🔥 ONNX エクスポート</summary>
<pre>
python dynamo.py export superpoint \
  --num-keypoints 1024 \
  -b 2 -h 1024 -w 1024 \
  -o weights/superpoint_lightglue_pipeline.onnx
</pre>
</details>

<details>
<summary>⚡ ONNX Runtime 推論 (CUDA)</summary>
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
<summary>🚀 ONNX Runtime 推論 (TensorRT)</summary>
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
<summary>🧩 TensorRT 推論</summary>
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
<summary>🟣 ONNX Runtime 推論 (OpenVINO)</summary>
<pre>
python dynamo.py infer \
  weights/superpoint_lightglue_pipeline.onnx \
  assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg \
  superpoint \
  -h 512 -w 512 \
  -d openvino
</pre>
</details>

## クレジット
もし本リポジトリのコードや論文のアイデアを使用した場合は、[LightGlue](https://arxiv.org/abs/2306.13643)、[SuperPoint](https://arxiv.org/abs/1712.07629)、および [DISK](https://arxiv.org/abs/2006.13566) の著者を引用することを検討してください。また、ONNX バージョンが役に立った場合は、このリポジトリにスターを付けていただけると幸いです。

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
  biburl       = {https://dblp.org/rec/journals/corr/abs-2006-13566.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
