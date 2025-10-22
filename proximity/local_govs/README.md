# 地方自治体のクラスタリング

[全国地方公共団体コード](https://www.soumu.go.jp/denshijiti/code.html)を持つ団体の **名称** から、
都道府県の **名称** に embedding ベクトルを用いてクラスタリングするタスク。
embedding には google/embeddinggemma-300m を用いる

大まかな手順

1. 都道府県名称を embedding でベクトル化
2. 団体名を embedding でベクトル化して、1のベクトル群でへクラスタリング
    *   `札幌市中央区` をそのまま使うか `中央区` とするかでバリエーションが作れる。
        後者がより難しい
3. accuracy を求める

前段のクラスタリング結果から学習データを作って学習し、再度計測し、accuracy の変化をみる。
