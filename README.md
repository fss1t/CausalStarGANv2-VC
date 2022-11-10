# CausalStarGANv2-VC
このリポジトリでは, 低遅延リアルタイムAny-to-Many声質変換に使用するCausalStarGANv2-VCモデルの訓練を行うスクリプトを公開しています. StarGANv2-VC[^1]および当手法で使用するJDCNet[^2]の非公式実装を含みます. CNNConformerではConformer[^3], CausalHiFi-GANではHiFi-GAN[^4]を利用しています. 

## 音声サンプル
[CausalStarGANv2-VC demo](https://fss1t.github.io/)に置いています.

## 使用方法
Python3.9以上で動作します.
### 1. datasetの配置
[JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)および[JVS-MuSiC](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music)からデータをコピーし, 再配置する.

`%path_jvs%`に[JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)のルートディレクトリ(jvs_ver1), `%path_jvs_corpus%`に[JVS-MuSiC](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music)のパスを指定し, 以下を実行する. 
```
python dataset/main.py --path_jvs %path_jvs% --path_jvs_music %path_jvs_music%
```

### 2. CausalHiFi-GAN, JDCNet, CNNConformerの学習
```
python CausalHiFiGAN/main.py
python JDCNet/main.py
python CNNConformer/main.py
```

### 3. StarGANv2-VCの学習
```
python StarGANv2VC/main.py
```
### 4. CausalStarGANv2-VCの学習
```
python CausalStarGANv2VC/main.py
```

## 参照

- CNNConformer/CNNConformer/models/conformer: https://github.com/sooftware/conformer
- CNNConformer/CNNConformer/models/cnn.py
- JDCNet/JDCNet/models/jdcnet.py
- StarGANv2VC/StarGANv2VC/models/*

  : https://github.com/yl4579/StarGANv2-VC

[^1]: StarGANv2-VC<br>
  paper: https://arxiv.org/abs/2107.10394#<br>
  official implementation: https://github.com/yl4579/StarGANv2-VC

[^2]: JDCNet<br>
  paper: https://www.mdpi.com/2076-3417/9/7/1324<br>
  official implementation: https://github.com/keums/melodyExtraction_JDC

[^3]: Conformer<br>
  paper: https://arxiv.org/abs/2005.08100

[^4]: HiFi-GAN<br>
  paper: https://arxiv.org/abs/2010.05646<br>
  official implementation: https://github.com/jik876/hifi-gan
