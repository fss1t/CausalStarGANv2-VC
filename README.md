# CausalStarGANv2-VC
このリポジトリは筆者が卒業研究で扱ったStarGANv2-VC[^1]および当手法で使用するJDCNet[^2]の**非公式**実装を含む就職活動用ポートフォリオである.
公開に際して[JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)および[JVS-MuSiC](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music)をデータセットとして学習および推論を行うデモを公開した.
## 使用方法
### 1. datasetの配置
[JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)および[JVS-MuSiC](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music)からデータをコピーし, 再配置する.

```
python dataset/main.py --path_jvs %path_jvs% --path_jvs_music %path_jvs_music%
```

### 2. CausalHiFiGANの学習


## 参照

- CNNConformer_ASR/common/conformer: https://github.com/sooftware/conformer
- CNNConformer_ASR/common/conformer/cnn.py
- JDCNet/common/model/jdcnet.py
- StarGANv2VC/common/model/models.py
- StarGANv2VC/train/tool/losses.py<br>
: https://github.com/yl4579/StarGANv2-VC

[^1]: StarGANv2-VC<br>
  paper: https://arxiv.org/abs/2107.10394#<br>
  official implementation: https://github.com/yl4579/StarGANv2-VC

[^2]: JDCNet<br>
  paper: https://www.mdpi.com/2076-3417/9/7/1324<br>
  official implementation: https://github.com/keums/melodyExtraction_JDC

[^3]: HiFi-GAN<br>
  paper: https://arxiv.org/abs/2010.05646<br>
  official implementation: https://github.com/jik876/hifi-gan
