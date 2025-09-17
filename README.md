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
