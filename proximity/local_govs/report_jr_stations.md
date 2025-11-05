# 派生課題: JR駅名のクラスタリング

城、旧国名に引き続き、JRの駅名(4263件)をクラスタリングしてみる

## 結果

駅名末尾に `駅` を含まないデータのクラスタリング

```
$ ./cluster_local_govs.py -l ./jr_stations.tsv -m google/embeddinggemma-300m
accuracy: 0.04167626064932074

$ ./cluster_local_govs.py -l ./jr_stations.tsv -m trained-lgov-full/checkpoint-100
accuracy: 0.2864379461201934

$ ./cluster_local_govs.py -l ./jr_stations.tsv -m trained-lgov-full/checkpoint-100
accuracy: 0.2965691918029012
```

駅名末尾に `駅` を含むデータのクラスタリング

```
$ ./cluster_local_govs.py -l ./jr_stations_eki.tsv -m google/embeddinggemma-300m
accuracy: 0.04351830531890399

$ ./cluster_local_govs.py -l ./jr_stations_eki.tsv -m trained-lgov-full/checkpoint-100
accuracy: 0.2958784250518075

$ ./cluster_local_govs.py -l ./jr_stations_eki.tsv -m trained-lgov-partial/checkpoint-100
accuracy: 0.2951876583007138
```

### 完全版

-   「駅」を含まない
    -   [未学習モデルによるクラスタリング結果](./result-jr_stations_0.txt)
    -   [地方自治体の完全名を学習したモデルによるクラスタリング結果](./result-jr_stations_1.txt)
    -   [地方自治体の部分名を学習したモデルによるクラスタリング結果](./result-jr_stations_2.txt)
-   「駅」を含む
    -   [未学習モデルによるクラスタリング結果](./result-jr_stations_eki_0.txt)
    -   [地方自治体の完全名を学習したモデルによるクラスタリング結果](./result-jr_stations_eki_1.txt)
    -   [地方自治体の部分名を学習したモデルによるクラスタリング結果](./result-jr_stations_eki_2.txt)

## 検討・考察

学習によりクラスタリング精度が約25ポイント上昇した。
この25ポイントは約1000件に相当する。
約1000件の駅名が地方自治体の名称を含む可能性がある。
旧国名のクラスタリングで発生したのと似たような現象が発生している可能性もある。

問い合わせ名に「駅」を付けることの影響はほぼない。
完全名学習モデルでは微改善、部分名学習モデルでは微改悪。
