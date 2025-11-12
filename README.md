# EmbeddingGemma

*   [EmbeddingGemma モデルの概要](https://ai.google.dev/gemma/docs/embeddinggemma?hl=ja)
*   [EmbeddingGemma @ Hugging Face](https://huggingface.co/collections/google/embeddinggemma-68b9ae3a72a82f0562a80dc4)
    *   300m
    *   300m-qat-q4\_0-unquantized
    *   300m-qat-q8\_0-unquantized

MTEB (Multilingual v2/English v2/Code v1)

若干(1.0~2.0)のスコア低下

MTEBとは? → Massive Text Embedding Benchmark

*   [テキスト埋め込みのベンチマークMTEB（Massive Text Embedding Benchmark）って？](https://kazuhira-r.hatenablog.com/entry/2024/01/06/011503)
*   [MTEB: Massive Text Embedding Benchmark @HF](https://huggingface.co/blog/mteb)
*   [日本語テキスト埋め込みベンチマークJMTEB（Japanese Massive Text Embedding Benchmark）](https://kazuhira-r.hatenablog.com/entry/2025/01/25/021323)
*   [リーダーボード](https://huggingface.co/spaces/mteb/leaderboard)
    サイズとしては良いが、0.6Bで高性能なモデル [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) もある

とりあえず…やってみるか

## First touch

セットアップはPythonを使ってこんな感じ。
Windowsネイティブの Python 3.13 と CUDA 12.8 を使っている。

```console
$ python -m venv venv
$ source ./venv/Scripts/activate
$ python -m pip install -U pip
$ python -m pip install sentence-transformers

# Windows + GPU で実行するのに必須
$ python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

[demo/embeddinggemma-reportcard.py](demo/embeddinggemma-reportcard.py)
このスクリプトは EmbeddingGemma のモデルカードのコードを改造したもの。

```
$ python ./demo/embeddinggemma-reportcard.py
(768,) (4, 768)
tensor([[0.4989, 0.7087, 0.5910, 0.5932]])
#1      7.220391035079956       import sentence_transformers
#2      4.993574380874634       load model
#3      1.1920928955078125e-06  define query and documents
#4      0.10607171058654785     embedding query
#5      0.09303450584411621     embedding documents
#6      0.00017380714416503906  print shape of embeddings
#7      0.00024628639221191406  calculate similarities
#8      0.002045154571533203    print similarities
```

embeddingの計算(#4, #5)はCPUを使って 0.1 秒程度とかなり速い。
ライブラリのロードとメモリのロードが12～13秒。

TorchをGPU版に入れ替え、GPU実行してみる。

[demo/embeddinggemma-reportcard-gpu.py](demo/embeddinggemma-reportcard-gpu.py)

```
$ python ./demo/embeddinggemma-reportcard-gpu.py
(768,) (4, 768)
tensor([[0.4989, 0.7087, 0.5910, 0.5932]])
#1      7.571403741836548       import sentence_transformers
#2      6.052325487136841       load model
#3      1.6689300537109375e-06  define query and documents
#4      0.2904388904571533      embedding query
#5      0.08441638946533203     embedding documents
#6      0.00011610984802246094  print shape of embeddings
#7      0.00028967857360839844  calculate similarities
#8      0.002015352249145508    print similarities
```

embeddingの初回実行である #4 はやや伸びた。
おそらくCUDAコードのGPUへのアップロード時間が入ってる。
一方で2回目は微減(10%未満)といったところ。
バッチサイズを増やせばまた結果は異なるかもしれないが、
GPUとのやり取りというオーバーヘッドを考慮すればCPUでも充分かもしれない。
モデルのロード時間はGPUへのアップロード分、伸びた。

QWen3-Embedding-0.6B で同じことをしてみる。
[demo/qwen3-embedding-0.6B.py](demo/qwen3-embedding-0.6B.py)

```
$ python ./demo/qwen3-embedding-0.6B.py
(1024,) (4, 1024)
tensor([[0.4811, 0.6901, 0.5838, 0.6643]])
#1      7.519515514373779       import sentence_transformers
#2      4.851632356643677       load model
#3      7.152557373046875e-07   define query and documents
#4      0.1429901123046875      embedding query
#5      0.2581138610839844      embedding documents
#6      0.00015926361083984375  print shape of embeddings
#7      0.00025582313537597656  calculate similarities
#8      0.0020155906677246094   print similarities
```

[demo/qwen3-embedding-0.6B-gpu.py](demo/qwen3-embedding-0.6B-gpu.py)

```
$ python ./demo/qwen3-embedding-0.6B-gpu.py
(1024,) (4, 1024)
tensor([[0.4811, 0.6901, 0.5838, 0.6643]])
#1      7.5174665451049805      import sentence_transformers
#2      5.719744682312012       load model
#3      1.430511474609375e-06   define query and documents
#4      0.3130626678466797      embedding query
#5      0.07213592529296875     embedding documents
#6      0.00014925003051757812  print shape of embeddings
#7      0.0003120899200439453   calculate similarities
#8      0.0020384788513183594   print similarities
```

### 類似度の検証

コード中のクエリ及び4例文は以下の通り。クエリと例文とのsimilarity 類似度を計算している。

クエリ:

> Which planet is known as the Red Planet?
> (訳: 赤い惑星として知られるのは?)

例文1:

> Venus is often called Earth's twin because of its similar size and proximity.
> (訳: 金星はそのサイズと近さから、しばしば地球の双子と呼ばれる)

例文2:

> Mars, known for its reddish appearance, is often referred to as the Red Planet.
> (訳: 火星はその赤い見た目から、しばしば赤い惑星として参照される)

例文3:

> Jupiter, the largest planet in our solar system, has a prominent red spot.
> (訳: 木星は、我々の太陽系で最も大きく、大赤斑がある)

例文4:

> Saturn, famous for its rings, is sometimes mistaken for the Red Planet.
> (訳: 土星は、輪があることで有名で、時々赤い惑星と間違われる)

正解は2で、3と4は紛らわしい。
特に4は **Red Planet** という語句を **be mistaken** で否定しているので難易度が高そう。

EmbeddingGemma と Qwen3 による類似度は以下の通り

| 例文 | EmbeddingGemma | Qwen3-Embedding-0.6B |
|-----:|---------------:|---------------------:|
|     1|         0.4989 |               0.4811 |
|     2|         0.7087 |               0.6901 |
|     3|         0.5910 |               0.5838 |
|     4|         0.5932 |               0.6643 |

両モデルに共通する特徴は以下の通り

*   ただしく正解である2に高いスコアを付けている
*   1と3では関連度の低い(紛らわしくない)、1のほうに低いスコアを付けている

一方でモデル間の違いとして

*   Qwen3が、特に紛らわしい4に対して、正解2との差が約0.03と高めのスコアを付けている。
    EmbeddingGemmaは正解2との差が約0.12と十分に低い。
*   次元数とベクトル要素の精度が異なる
    *   EmbeddingGemma 768 (float32) (3072 bytes) : ベクトルサイズは 128,256,512,768 で可変
    *   Qwen3-Embedding 1024 (float32) (4096 bytes) : モデルの重みは BF16 なので実は半分でも良いのでは?

### プロンプトによる指示

プロンプトprefixで、タスク毎に最適な embeddings を得られる。
例:

*   `task: search result | query: {content}`
*   `title: {title | "none"} | text: {content}`
*   `task: question answering | query: {content}`
*   `task: fact checking | query: {content}`
*   `task: classification | query: {content}`
*   `task: task: clustering | query: {content}`
*   `task: sentence similarity | query: {content}`
*   `task: code retrieval | query: {content}`

デフォルトは `search result` タスク

### 疑問点

*   より長いテキストはどう食わせる?

    たぶんなんか良い、一般的な方法がある

*   学習(finetune)はどうやる?

    入力と学習に使う差分量の計算がわからない

### 学習(Finetune)はどうやるの?

[EmbeddingGemma をファインチューニングする](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers?hl=ja)

> (アンカー、ポジティブ、ネガティブ) の 3 つ組として構造化されます

> EmbeddingGemma で最適なエンベディングを生成するには、入力テキストの先頭に「指示プロンプト」または「タスク」を追加する必要があります。文の類似性には STS を使用します。

ここでいう `STS` は `sentence similarity` だと推測される。

```
def get_scores(query, documents):
  # Calculate embeddings by calling model.encode()
  query_embeddings = model.encode(query, prompt=task_name)
  doc_embeddings = model.encode(documents, prompt=task_name)

  # Calculate the embedding similarities
  similarities = model.similarity(query_embeddings, doc_embeddings)
```

`model.encode_document` と `mode.encode(..., prompt=task_name)` は別物であることに注意が必要

> If you are unsure whether you should use encode(), encode_query(), or
> encode_document(), your best bet is to use encode_query() and
> encode_document() for Information Retrieval tasks with clear query and
> document/passage distinction, and use encode() for all other tasks.

from: <https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html>

`encode_query` と `encode_document` を使い分けたほうが良い、と

`SentenceTransformerTrainingArguments` と `SentenceTransformerTrainer` で学習

## Second touch

目標

*   速度ベンチマーク手段を確立
    *   GPUとCPUの違い
    *   モデルの違い(候補: QWen3-Embedding-0.6B)
*   日本語における評価
    *   JMTEBを検討
*   実行にかかる各種スペック計測
    *   メモリ量
    *   マルチスレッド性能
    *   速度(上記ベンチマークで対応可能)
*   MTEBのタスクでやっていることを理解する
    *   Banking77Classification
    *   GPUSpeedTask
    *   CPUSpeedTask
    *   JSICK
    *   SICK-R
    *   JSTS
    *   STS16

実行環境は例によって i9-9900K + RTX4070 12GB

### pip install

```console
$ pip install mteb
$ pip install mteb[speedtask]
```

### MTEB

<https://github.com/embeddings-benchmark/mteb>

とりあえず一旦 EmbeddedGemma で実行してみる。
タスクマネージャーを見る限り、GPUで実行されているようだ。

実行するとタスク毎モデル毎に results にファイルが出力される

以下、実行結果。分類タスクの成績も出ているがまずは速度に着目したいので、そこだけ切り抜く。

``` console
$ mteb run -m 'google/embeddinggemma-300m' -t Banking77Classification --verbosity 3
...(snipped)...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.
...(snipped)...
INFO:mteb.evaluation.MTEB:Evaluation for Banking77Classification on test took 66.60 seconds
```

GPUメモリ使用量は2.3GB

QWen3-Embedding-0.6B でも同じことをする。
GPUメモリ使用量は4.1GB程度。

```console
$ mteb run -m 'QWen/QWen3-Embedding-0.6B' -t Banking77Classification --verbosity 3


...(snipped)...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.
...(snipped)...
INFO:mteb.evaluation.MTEB:Evaluation for Banking77Classification on test took 68.01 seconds
```

実行速度は誤差程度。
モデルの違いによる実行速度への影響は大きくなさそう。

GPUメモリは実行前に0.6GB使ってる。
そのため実質の使用量はEG (EmbeddedGemma) が 1.7GB、
QE (Qwen3 Embedding) が 3.5GB と概算できる。
重みは QE が EG のほぼ倍なのでそれが反映されていると考えて良さそう。

EGベースで考えると1.7GB/0.3GB なのでおよそ x6 のGPUメモリを使ってそう。
ログからx16のconcurrencyで動いていると推定される。
そのx16分のメモリがどう計上されているかは不明。

別のタスクとして SpeedTask カテゴリに CPUSpeedTask と GPUSpeedTask があるので試してみる。

```console
$ mteb run -m 'google/embeddinggemma-300m' -t GPUSpeedTask --verbosity 2
...(snipped)...
INFO:mteb.evaluation.MTEB:Evaluation for GPUSpeedTask on test took 3.79 seconds

$ mteb run -m 'google/embeddinggemma-300m' -t CPUSpeedTask --verbosity 2
...(snipped)...
INFO:mteb.evaluation.MTEB:Evaluation for CPUSpeedTask on test took 59.69 seconds
```

約15.7倍の差が付いた。充分早いとは思うが、GPUを使うべきではある結果。

QE との差もみておく。
約2.54倍の差が付いた。
やはりモデルが小さい分、速度は速い用だ。
QEをCPUで実行するのはやめておくが、予想では2～3分かかりそう。

```
$ mteb run -m 'QWeb3/QWeb3-Embedding-0.6B' -t GPUSpeedTask --verbosity 2
...(snipped)...
INFO:mteb.evaluation.MTEB:Evaluation for GPUSpeedTask on test took 9.63 seconds
```

### 対日本語

mteb に JSICK 及び JSTS というタスクが含まれる。
これらの英語版と比べてみよう。
とりあえず JSICK とその英語版らしき SICK-R を実行。
SICK-Rのほうが例文数が多いのか、時間がかかる。

```console
# Windows用Pythonで、UTF-8をデフォルトエンコーディングにする
$ export PYTHONUTF8=1

$ mteb run -m 'google/embeddinggemma-300m' -t JSICK

$ cat results/google__embeddinggemma-300m/no_revision_available/JSICK.json
{
  "dataset_revision": "e4af6c73182bebb41d94cb336846e5a452454ea7",
  "task_name": "JSICK",
  "mteb_version": "1.36.15",
  "scores": {
    "test": [
      {
        "pearson": 0.669638,
        "spearman": 0.66909,
        "cosine_pearson": 0.669638,
        "cosine_spearman": 0.66909,
        "manhattan_pearson": 0.662281,
        "manhattan_spearman": 0.669476,
        "euclidean_pearson": 0.66218,
        "euclidean_spearman": 0.66909,
        "main_score": 0.66909,
        "hf_subset": "default",
        "languages": [
          "jpn-Jpan"
        ]
      }
    ]
  },
  "evaluation_time": 7.843804121017456,
  "kg_co2_emissions": null
}

$ mteb run -m 'google/embeddinggemma-300m' -t SICK-R

$ cat results/google__embeddinggemma-300m/no_revision_available/SICK-R.json
{
  "dataset_revision": "20a6d6f312dd54037fe07a32d58e5e168867909d",
  "task_name": "SICK-R",
  "mteb_version": "1.36.15",
  "scores": {
    "test": [
      {
        "pearson": 0.594934,
        "spearman": 0.570358,
        "cosine_pearson": 0.594934,
        "cosine_spearman": 0.570358,
        "manhattan_pearson": 0.582721,
        "manhattan_spearman": 0.569466,
        "euclidean_pearson": 0.583439,
        "euclidean_spearman": 0.570358,
        "main_score": 0.570358,
        "hf_subset": "default",
        "languages": [
          "eng-Latn"
        ]
      }
    ]
  },
  "evaluation_time": 37.59281349182129,
  "kg_co2_emissions": null
}
```

JSICKのほうが点数が良いが…なんかログからは、
事前に学習している感じもする。
ベンチマークの内容を調べる必要もありそう。

念のため QE でも実行しておく。

```console
$ mteb run -m 'QWen/QWen3-Embedding-0.6B' -t JSICK

$ mteb run -m 'QWen/QWen3-Embedding-0.6B' -t SICK-R
```

その結果へのリンク。スコアは双方0.8を超え、かなり高い。

* [QE/JSICK result](./results/QWen__QWen3-Embedding-0.6B/no_revision_available/JSICK.json)
* [QE/SICK-R result](./results/QWen__QWen3-Embedding-0.6B/no_revision_available/SICK-R.json)

次にJSTSを実行してみる。

```console
$ mteb run -m 'google/embeddinggemma-300m' -t JSTS

$ mteb run -m 'QWen/QWen3-Embedding-0.6B' -t JSTS
```

が、データセットが消えてて実行できない。
yahoojapan/JGLUE がソースなのだが jsts-v1.3 にアップデートして v1.1 を消したみたいだ。

<https://huggingface.co/datasets/shunk031/JGLUE> が間に入ってる。
こいつがGitHub上のファイルを Commit ID やタグベースではなく
main ブランチベースで指定しているのが原因らしい。

~/.cache/hugging-face/datasets/downloads 内の該当しそうな箇所を以下のURLに訂正して実行してみる。一応、実行できた。

*   <https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.1.0/datasets/jsts-v1.1/train-v1.1.json>
*   <https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.1.0/datasets/jsts-v1.1/valid-v1.1.json>

その結果:

*   [EB/JSTS result](./results/google__embeddinggemma-300m/no_revision_available/JSTS.json)
*   [QE/JSTS result](./results/QWen__QWen3-Embedding-0.6B/no_revision_available/JSTS.json)

おおよそJSICKと変わらないスコア傾向。

STS16STS も実行してみる。

```console
$ mteb run -m 'google/embeddinggemma-300m' -t STS16

$ mteb run -m 'QWen/QWen3-Embedding-0.6B' -t STS16
```

*   [EB/STS16 result](./results/google__embeddinggemma-300m/no_revision_available/STS16.json)
*   [QE/STS16 result](./results/QWen__QWen3-Embedding-0.6B/no_revision_available/STS16.json)

### ベンチマークの中身

SpeeedTaskから。実装は [AbsTaskSpeedTask](https://github.com/embeddings-benchmark/mteb/blob/6e72dc0e30577c2fde2bb5c6cb55c79bf8e1eaec/mteb/abstasks/AbsTaskSpeedTask.py#L22)

43のデータを7回 encode してかかった時間を記録して統計データを取っている。
EBとQEで中央値でEBのほうが3倍速い。

JSICKとSICK-R はどちらも [AbsTaskSTS](https://github.com/embeddings-benchmark/mteb/blob/6e72dc0e30577c2fde2bb5c6cb55c79bf8e1eaec/mteb/abstasks/AbsTaskSTS.py) ベースで、データ数が違うだけ。
評価は [STSEvaluator](https://github.com/embeddings-benchmark/mteb/blob/6e72dc0e30577c2fde2bb5c6cb55c79bf8e1eaec/mteb/evaluation/evaluators/STSEvaluator.py) でやってる。
ペアとなる文章のembeddingを求めて、その間の複数の距離を求めてる。
STS (Semantic Textual Similarity) そんな感じか。

JSICKの生データ <https://github.com/verypluming/JSICK>

embeddingの距離と実際のスコアとの関係を、以下の相関関数で評価している。

*   [ピアソンの積率相関係数](https://ja.wikipedia.org/wiki/%E3%83%94%E3%82%A2%E3%82%BD%E3%83%B3%E3%81%AE%E7%A9%8D%E7%8E%87%E7%9B%B8%E9%96%A2%E4%BF%82%E6%95%B0)
*   [スピアマンの順位相関係数](https://ja.wikipedia.org/wiki/%E3%82%B9%E3%83%94%E3%82%A2%E3%83%9E%E3%83%B3%E3%81%AE%E9%A0%86%E4%BD%8D%E7%9B%B8%E9%96%A2%E4%BF%82%E6%95%B0)

Banking77Classification は AbsTaskClassification で中身は kNNClassificationEvaluator ということはほぼほぼ k-NN 要素。embeddingで与えられた距離でk-NNして、理想にどれだけ近いかを測ってる。

### Second touchのまとめ

*   GPUはCPUの約16倍速いので可能ならGPUを使うべし
*   精度は QWen3-Embedding-0.6B の方が良いが、パフォーマンスはEmbedding Gemmaのほうが約2.5倍良い。メモリ量はQWen3は2倍
*   日本語だから特に良い or 悪いという傾向はみられなかった。データセット依存の部分が大きいので、データセットの言語の違いを検証ででいるほどではない
*   MTEBはモデル間の性能を見るモノ
*   FP16で0.6GB vs 1.2GB になり、さらに x3 ぐらいかな?

## Third touch

*   EmbeddingGemmaを実アプリで使うことを考える
    *   llama.cpp で良いんじゃない?
    *   [Java wrapper](https://github.com/kherud/java-llama.cpp) もあるけど、実用的じゃないかも
    *   実行方法やパフォーマンスは?

llama.cpp は b6653 が最新だった。

GGUFフォーマットモデルは [ggml-org/embeddinggemma-300M-GGUF](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) で良さそう。
シンプルな使い方が記載されている。

```
llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings
```

モデルはWindowsだと `AppData\Local\llama.cpp` にダウンロードされた。
また自動でGPUを使うようにロードされた。

```
curl --request POST \
    --url http://localhost:8080/embedding \
    --header "Content-Type: application/json" \
    --data '{"input": "Hello embeddings"}' \
    --silent
```

アクセスは [OpenAPI の Embeddings endpoint](https://platform.openai.com/docs/api-reference/embeddings) にほぼ準拠するらしい。
`input` 要素に文字列もしくは文字列の配列を受け付ける。

### パフォーマンス測定

MTEB の [SpeedTask の例文](https://github.com/embeddings-benchmark/mteb/blob/6e72dc0e30577c2fde2bb5c6cb55c79bf8e1eaec/mteb/abstasks/the_ugly_duckling.txt)と同等の[入力を作成](the_ugly_duckling.json)し、それでパフォーマンスを測定してみる。

```console
$ time curl -X POST -d @the_ugly_duckling.json -H 'Content-Type: application/json' --silent http://localhost:8080/embedding -o tmp/out.json

real    0m0.772s
user    0m0.015s
sys     0m0.015s
```

1回の実行にかかる時間は0.77～0.81秒。
MTEBの方は7回実行して約3.8秒なので、1回当たり約0.54。
最大0.3秒くらいのオーバーヘッドがある換算だが、まぁそんなもんかな?という感じ。

25行、50行、100行のデータを作って計測してみる。
各データごとに5回づつ計測。単位は秒

|    | 25    | 50    | 100 
|----|------:|------:|------:
|#1  | 0.496 | 0.716 | 1.146
|#2  | 0.484 | 0.689 | 1.144
|#3  | 0.497 | 0.704 | 1.125
|#4  | 0.490 | 0.686 | 1.116
|#5  | 0.480 | 0.741 | 1.161
|Avg.| 0.489 | 0.707 | 1.138

どうもワームアップがあるようで間隔を短くして投げると速いかも?
元々の43行のデータは0.64～0.67秒くらいになる。
0.1秒のオーバーヘッド考えると納得度は増す。

行数に対する応答性はほぼ線形。
約0.28秒の固定オーバーヘッド + 1行あたり 約8.4ms くらい。

CPUでも同じことをやってみると… 43行で約4.678秒。
GPUの約8倍くらいの時間がかかっている。
MTEBのCPUSpeedTaskよりは速いが、llama.cppのCPU実装が優秀という理解で良いだろう。

CPUは25行で2.95秒、50行で5.30秒くらい。 
約0.6秒の固定オーバーヘッドに1行あたり約 94ms といったところ。

### キャッシュ

<https://github.com/ggml-org/llama.cpp/pull/7826>

- `$LLAMA_CACHE` if defined,
- `~/Library/Caches/llama.cpp/` on Mac,
- `~/.cache/llama.cpp` on Linux (or `$XDG_CACHE_HOME/llama.cpp` if defined)
- `%LOCALAPPDATA%\llama.cpp` on Windows

ということで環境変数 `$LLAMA_CACHE` を設定すれば任意の場所に変更できそう。

## Fourth touch

[./proximity](./proximity) で実験中。

[都道府県庁所在地の位置](https://gist.github.com/ctsaran/42728dad3c7d8bd91f1d) を教師データとし、
都道府県名から得た embedding 同士間の距離のTop-K (近い順)を比較する。

物理的に近いもの同士が embedding 間でも近くなるように学習を掛ける。
その学習前後の Top-K accuracy を比較して評価するという算段。

embeddingの計算には clustering タスクを用いた。
プロンプトは `task: clustering | query: `

ユークリッド距離での計算。コサイン距離は未確認。
未学習の状態では、特に字面が近いものが近傍になるようだった。
また概念的な近さも一部考慮されることがあるようだ。
例えば北海道と沖縄が近いなど。

./train.py で学習。 pref-embedding-gemma を出力。
`pip install 'accelerate>=0.26.0` で学習用のモジュールのインストールが必要。

評価は Top-10 の accuracy (10個の内、何個正解したか) で求める。

accuracyは学習前後で 0.23 から 0.72 に改善した。
0.23というのは 47 都道府県から適当に10選んだ時の値が 0.21 だから、あてずっぽうと変わらない。

* [学習前 accuracy](./proximity/accuracy_0.txt)
* [学習後 accuracy_1](./proximity/accuracy_1.txt)


しかし市区町村や住所のTop-Kは学習後のほうが悪くなった。
下手に学習させたことで元々持っていたクラスタリング能力を失ったと考えられる。

* [学習前 Top-K のクエリー](./proximity/topk_queries0.log)
* [学習後 Top-K のクエリー](./proximity/topk_queries1.log)

ある意味で学習が進んだことを示しているが、都道府県間の距離関係を学習させるのは有用な学習ではない。

次は「市区町村(自治体)名を都道府県でクラスタリングする」という問題設定で検証すべきかもしれない。
ただそうなると学習の実験ではなく、モデルの妥当性検証に相当する。
もしくはタスクプロンプトを変えると言うことでも良いかも。

## Prompts in the models

モデルに組み込まれたプロンプトを確認した。

* [EmbeddingGemma 300M](./results/prompts/embeddinggemma-300m.tsv)
* [Qwen3 Embedding 0.6B](./results/prompts/Qwen3-Embedding-0.6B.tsv)

```console
$ ./bin/show_prompts.py > ./results/prompts/embeddinggemma-300m.tsv

$ ./bin/show_prompts.py -m Qwen/Qwen3-Embedding-0.6B > ./results/prompts/Qwen3-Embedding-0.6B.tsv
```
