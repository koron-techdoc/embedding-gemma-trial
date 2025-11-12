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

## クラスタリングの派生課題

-   [城のクラスタリング結果](./report_castles.md)
-   [旧国名のクラスタリング結果](./report_old_countries.md)
-   [JR駅名のクラスタリング結果](./report_jr_stations.md)

## Qwen3 Embedding 0.6B で同じことをやる

Qwen3 Embedding 0.6B (以下Qwen3) は、queryとdocumentの2タスクしかないので、document を採用。
即ちプレフィックスプロンプトは無しで単に自治体名でベクトルを生成する。

### 実験内容と結果

1. 未学習の状態で一旦クラスタリング

        ./cluster_local_govs.py -m QWen/Qwen3-Embedding-0.6B -k document > ./QWen3/pretrained-accuracy.txt

    [学習前のクラスタリング結果](./Qwen3/pretrained-accuracy.txt)
2. 学習用データ作成

        go run ./gen_train.go ./QWen3/pretrained-accuracy.txt > ./QWen3/train.tsv

    [学習用データの詳細](./Qwen3/train.tsv)

3. 学習を実行

        ./train.py -m QWen/Qwen3-Embedding-0.6B -k document -t ./QWen3/train.tsv -o ./QWen3/trained_model -b 25

    モデルが大きいため学習バッチサイズをembeddinggemmna-300mの1/4にした。

4. 学習の進行度の評価

        ./cluster_local_govs.py -m ./QWen3/trained_model/checkpoint-385 -k document > ./QWen3/accuracy-trained-100.txt
        ./cluster_local_govs.py -m ./QWen3/trained_model/checkpoint-308 -k document > ./QWen3/accuracy-trained-080.txt
        ./cluster_local_govs.py -m ./QWen3/trained_model/checkpoint-231 -k document > ./QWen3/accuracy-trained-060.txt
        ./cluster_local_govs.py -m ./QWen3/trained_model/checkpoint-154 -k document > ./QWen3/accuracy-trained-040.txt
        ./cluster_local_govs.py -m ./QWen3/trained_model/checkpoint-77  -k document > ./QWen3/accuracy-trained-020.txt

    学習途中のモデルに学習データをクラスタリングさせaccuracyを求め、学習が如何に進んだかを評価する。

    | 学習率 | Accuracy | 結果ファイル |
    |-------:|---------:|--------------|
    | 0%     | 0.080    | <http:./QWen3/pretrained-accuracy.txt>  |
    | 20%    | 0.273    | <http:./QWen3/accuracy-trained-020.txt> |
    | 40%    | 0.522    | <http:./QWen3/accuracy-trained-040.txt> |
    | 60%    | 0.816    | <http:./QWen3/accuracy-trained-060.txt> |
    | 80%    | 0.937    | <http:./QWen3/accuracy-trained-080.txt> |
    | 100%   | 0.969    | <http:./QWen3/accuracy-trained-100.txt> |

    参考のために EmbeddingGemma 300M での学習進行をまとめると以下の通り

    | 学習率 | Accuracy | 結果ファイル |
    |-------:|---------:|--------------|
    | 0%     | 0.128    | <http:./accuracy_full.txt>  |
    | 20%    | 0.390    | <http:./accuracy_full_trained-020.txt> |
    | 40%    | 0.568    | <http:./accuracy_full_trained-040.txt> |
    | 60%    | 0.783    | <http:./accuracy_full_trained-060.txt> |
    | 80%    | 0.917    | <http:./accuracy_full_trained-080.txt> |
    | 100%   | 0.955    | <http:./accuracy_full_trained-100.txt> |

5.  クラスタリングの派生課題

    1. 城のクラスタリグ

        学習前後でのAccuracyの変化: 0.083 → 0.750

        参考: EmbeddingGemmaでの変化: 0.083 → 0.833

        <details>
        <summary>クラスタリングの詳細</summary>

        学習後

        ```
        $ ./cluster_local_govs.py -m ./QWen3/trained_model/checkpoint-385 -k document -l ./japan_castles.tsv
        弘前城  1/1     青森県  岩手県  青森県
        松本城  0/1     熊本県  愛媛県  長野県
        丸岡城  0/1     福岡県  宮城県  福井県
        犬山城  1/1     愛知県  愛媛県  愛知県
        彦根城  1/1     滋賀県  栃木県  滋賀県
        姫路城  1/1     兵庫県  大分県  兵庫県
        松江城  1/1     島根県  滋賀県  島根県
        備中松山城      0/1     愛媛県  福島県  岡山県
        丸亀城  1/1     香川県  山梨県  香川県
        伊予松山城      1/1     愛媛県  滋賀県  愛媛県
        宇和島城        1/1     愛媛県  鹿児島県        愛媛県
        高知城  1/1     高知県  滋賀県  高知県

        accuracy: 0.75
        ```

        学習前

        ```
        $ ./cluster_local_govs.py -m QWen/Qwen3-Embedding-0.6B -k document -l ./japan_castles.tsv
        弘前城  0/1     宮城県  岐阜県  青森県
        松本城  0/1     熊本県  宮城県  長野県
        丸岡城  0/1     宮城県  広島県  福井県
        犬山城  0/1     岐阜県  鳥取県  愛知県
        彦根城  0/1     京都府  宮城県  滋賀県
        姫路城  0/1     宮城県  東京都  兵庫県
        松江城  0/1     東京都  宮崎県  島根県
        備中松山城      0/1     宮城県  広島県  岡山県
        丸亀城  0/1     宮城県  広島県  香川県
        伊予松山城      0/1     宮城県  岐阜県  愛媛県
        宇和島城        0/1     広島県  鹿児島県        愛媛県
        高知城  1/1     高知県  宮城県  高知県

        accuracy: 0.08333333333333333
        ```

        </details>

    2. 旧国名のクラスタリング

        学習前後でのAccuracyの変化: 0.035 → 0.529

        参考: EmbeddingGemmaでの変化: 0.035 → 0.647

        <details>
        <summary>クラスタリングの詳細</summary>

        学習後

        ```
        $ ./cluster_local_govs.py -m ./QWen3/trained_model/checkpoint-385 -k document -l ./japan_old_countries.tsv
        陸奥    0/1     岩手県  岐阜県  青森県
        陸中    0/2     島根県,広島県   愛媛県,群馬県   岩手県,秋田県
        陸前    0/1     岩手県  青森県  宮城県
        羽後    1/2     秋田県,栃木県   鳥取県,宮城県   秋田県,山形県
        羽前    0/1     大阪府  宮城県  山形県
        岩代    0/1     岩手県  茨城県  福島県
        磐城    0/1     福岡県  宮城県  福島県
        常陸    1/1     茨城県  徳島県  茨城県
        下総    0/2     和歌山県,岡山県 岩手県,岐阜県   茨城県,千葉県
        下野    1/1     栃木県  静岡県  栃木県
        上野    0/1     栃木県  山梨県  群馬県
        武蔵    1/3     東京都,静岡県,兵庫県    茨城県,神奈川県,大分県  埼玉県,東京都,神奈川県
        上総    0/1     和歌山県        長野県  千葉県
        安房    0/1     沖縄県  山口県  千葉県
        相模    1/1     神奈川県        香川県  神奈川県
        越後    0/1     福井県  群馬県  新潟県
        佐渡    1/1     新潟県  高知県  新潟県
        越中    0/1     高知県  島根県  富山県
        能登    1/1     石川県  滋賀県  石川県
        加賀    1/1     石川県  栃木県  石川県
        越前    1/1     福井県  岐阜県  福井県
        若狭    1/1     福井県  新潟県  福井県
        甲斐    1/1     山梨県  熊本県  山梨県
        信濃    1/1     長野県  群馬県  長野県
        美濃    1/1     岐阜県  和歌山県        岐阜県
        飛騨    1/1     岐阜県  岩手県  岐阜県
        駿河    0/1     奈良県  茨城県  静岡県
        伊豆    1/1     静岡県  兵庫県  静岡県
        遠江    0/1     滋賀県  北海道  静岡県
        尾張    1/1     愛知県  愛媛県  愛知県
        三河    0/1     山形県  島根県  愛知県
        伊勢    1/1     三重県  福島県  三重県
        伊賀    1/1     三重県  福島県  三重県
        志摩    1/1     三重県  石川県  三重県
        紀伊    1/2     京都府,三重県   岐阜県,東京都   三重県,和歌山県
        近江    1/1     滋賀県  石川県  滋賀県
        山城    0/1     山梨県  宮城県  京都府
        丹波    1/2     兵庫県,山梨県   宮崎県,石川県   京都府,兵庫県
        丹後    0/1     北海道  栃木県  京都府
        摂津    2/2     大阪府,兵庫県   静岡県,神奈川県 大阪府,兵庫県
        和泉    1/1     大阪府  京都府  大阪府
        河内    0/1     茨城県  大阪府  大阪府
        播磨    1/1     兵庫県  大阪府  兵庫県
        但馬    0/1     群馬県  沖縄県  兵庫県
        淡路    1/1     兵庫県  大分県  兵庫県
        大和    0/1     宮城県  奈良県  奈良県
        因幡    0/1     三重県  岐阜県  鳥取県
        伯耆    1/1     鳥取県  滋賀県  鳥取県
        石見    0/1     北海道  富山県  島根県
        出雲    1/1     島根県  和歌山県        島根県
        隠岐    1/1     島根県  鳥取県  島根県
        備前    1/2     岡山県,福岡県   群馬県,沖縄県   岡山県,香川県
        備中    0/1     群馬県  徳島県  岡山県
        美作    1/1     岡山県  沖縄県  岡山県
        備後    0/1     福岡県  群馬県  広島県
        安芸    0/1     高知県  広島県  広島県
        周防    1/1     山口県  高知県  山口県
        長門    1/1     山口県  熊本県  山口県
        阿波    1/1     徳島県  長崎県  徳島県
        讃岐    0/1     福岡県  長崎県  香川県
        伊予    1/1     愛媛県  沖縄県  愛媛県
        土佐    1/1     高知県  広島県  高知県
        筑前    1/1     福岡県  岩手県  福岡県
        筑後    1/1     福岡県  岩手県  福岡県
        豊前    1/2     福岡県,沖縄県   愛知県,愛媛県   福岡県,大分県
        肥前    0/2     静岡県,愛媛県   福岡県,愛知県   佐賀県,長崎県
        壱岐    1/1     長崎県  鹿児島県        長崎県
        対馬    1/1     長崎県  徳島県  長崎県
        肥後    0/1     北海道  福岡県  熊本県
        豊後    0/1     福岡県  愛媛県  大分県
        日向    1/1     宮崎県  佐賀県  宮崎県
        薩摩    1/1     鹿児島県        熊本県  鹿児島県
        大隅    1/1     鹿児島県        長崎県  鹿児島県
        琉球    0/1     鹿児島県        福岡県  沖縄県

        accuracy: 0.5294117647058824
        ```

        学習前

        ```
        $ ./cluster_local_govs.py -m QWen/Qwen3-Embedding-0.6B -k document -l ./japan_old_countries.tsv
        陸奥    0/1     福島県  北海道  青森県
        陸中    0/2     広島県,福島県   東京都,兵庫県   岩手県,秋田県
        陸前    0/1     兵庫県  富山県  宮城県
        羽後    0/2     福島県,兵庫県   福岡県,広島県   秋田県,山形県
        羽前    0/1     鳥取県  大阪府  山形県
        岩代    0/1     岩手県  広島県  福島県
        磐城    0/1     宮城県  広島県  福島県
        常陸    0/1     福島県  兵庫県  茨城県
        下総    0/2     東京都,広島県   福島県,新潟県   茨城県,千葉県
        下野    0/1     長野県  新潟県  栃木県
        上野    0/1     長野県  京都府  群馬県
        武蔵    1/3     広島県,兵庫県,東京都    奈良県,京都府,福島県    埼玉県,東京都,神奈川県
        上総    0/1     東京都  広島県  千葉県
        安房    0/1     福島県  岐阜県  千葉県
        相模    0/1     山形県  京都府  神奈川県
        越後    0/1     福島県  新潟県  新潟県
        佐渡    0/1     福島県  広島県  新潟県
        越中    0/1     新潟県  京都府  富山県
        能登    0/1     北海道  神奈川県        石川県
        加賀    0/1     滋賀県  兵庫県  石川県
        越前    0/1     富山県  新潟県  福井県
        若狭    0/1     北海道  神奈川県        福井県
        甲斐    0/1     岐阜県  富山県  山梨県
        信濃    0/1     広島県  福島県  長野県
        美濃    0/1     福島県  東京都  岐阜県
        飛騨    0/1     奈良県  福島県  岐阜県
        駿河    0/1     広島県  福島県  静岡県
        伊豆    0/1     北海道  島根県  静岡県
        遠江    0/1     福島県  広島県  静岡県
        尾張    0/1     福島県  兵庫県  愛知県
        三河    0/1     京都府  三重県  愛知県
        伊勢    0/1     福島県  島根県  三重県
        伊賀    0/1     愛知県  滋賀県  三重県
        志摩    0/1     神奈川県        高知県  三重県
        紀伊    0/2     広島県,福島県   東京都,兵庫県   三重県,和歌山県
        近江    0/1     京都府  新潟県  滋賀県
        山城    0/1     宮城県  山形県  京都府
        丹波    0/2     大阪府,熊本県   岐阜県,京都府   京都府,兵庫県
        丹後    0/1     福島県  福井県  京都府
        摂津    0/2     茨城県,神奈川県 秋田県,福島県   大阪府,兵庫県
        和泉    0/1     神奈川県        和歌山県        大阪府
        河内    0/1     東京都  京都府  大阪府
        播磨    0/1     秋田県  香川県  兵庫県
        但馬    0/1     群馬県  兵庫県  兵庫県
        淡路    0/1     富山県  福井県  兵庫県
        大和    0/1     広島県  福島県  奈良県
        因幡    0/1     岐阜県  福井県  鳥取県
        伯耆    0/1     岐阜県  滋賀県  鳥取県
        石見    0/1     福島県  宮崎県  島根県
        出雲    0/1     福島県  島根県  島根県
        隠岐    0/1     福島県  岐阜県  島根県
        備前    0/2     宮城県,広島県   東京都,福島県   岡山県,香川県
        備中    0/1     兵庫県  東京都  岡山県
        美作    0/1     神奈川県        香川県  岡山県
        備後    1/1     広島県  福島県  広島県
        安芸    0/1     山梨県  秋田県  広島県
        周防    0/1     広島県  宮崎県  山口県
        長門    0/1     福島県  宮崎県  山口県
        阿波    0/1     福島県  大阪府  徳島県
        讃岐    0/1     福島県  茨城県  香川県
        伊予    0/1     岐阜県  福島県  愛媛県
        土佐    0/1     福井県  新潟県  高知県
        筑前    0/1     茨城県  山口県  福岡県
        筑後    0/1     兵庫県  福島県  福岡県
        豊前    0/2     福島県,奈良県   岐阜県,富山県   福岡県,大分県
        肥前    0/2     福井県,秋田県   富山県,茨城県   佐賀県,長崎県
        壱岐    0/1     岐阜県  福島県  長崎県
        対馬    0/1     広島県  福島県  長崎県
        肥後    0/1     福井県  福島県  熊本県
        豊後    0/1     福島県  富山県  大分県
        日向    0/1     福島県  広島県  宮崎県
        薩摩    0/1     福島県  島根県  鹿児島県
        大隅    0/1     北海道  富山県  鹿児島県
        琉球    1/1     沖縄県  北海道  沖縄県

        accuracy: 0.03529411764705882
        ```

        </details>

    3. JR駅名のクラスタリング

        Accuracy の変化

        | 駅名表記の方法  | 学習前 | 学習後 |
        |-----------------|-------:|-------:|
        |「駅」を含まない | 0.047  | 0.249  |
        |「駅」を含む     | 0.046  | 0.224  |

        (参考) EmbeddingGemma における Accuracy の変化

        | 駅名表記の方法  | 学習前 | 学習後 |
        |-----------------|-------:|-------:|
        |「駅」を含まない | 0.042  | 0.286  |
        |「駅」を含む     | 0.043  | 0.296  |

        実行したコマンド: 

        ```
        $ ./cluster_local_govs.py -m ./QWen3/trained_model/checkpoint-385 -k document -l ./jr_stations.tsv > ./QWen3/result-jr_stations.txt

        $ ./cluster_local_govs.py -m QWen/Qwen3-Embedding-0.6B -k document -l ./jr_stations.tsv > ./QWen3/result-jr_stations-0.txt

        $ ./cluster_local_govs.py -m ./QWen3/trained_model/checkpoint-385 -k document -l ./jr_stations_eki.tsv > ./QWen3/result-jr_stations+eki.txt

        $ ./cluster_local_govs.py -m QWen/Qwen3-Embedding-0.6B -k document -l ./jr_stations_eki.tsv > ./QWen3/result-jr_stations+eki-0.txt
        ```

        結果へのリンク

        * <http:./QWen3/result-jr_stations.txt>
        * <http:./QWen3/result-jr_stations-0.txt>
        * <http:./QWen3/result-jr_stations+eki.txt>
        * <http:./QWen3/result-jr_stations+eki-0.txt>

### 検討・考察

EmbeddingGemmaと比較して、
素の状態ではQWen3のほうが性能が 0.04 悪い。
学習後の学習データへの正確性は QWen3 のほうが  0.01 良くなる。

しかし学習後モデルへのクラスタリング派生課題に対しては、最大 0.1 ほど EmbeddingGemma の方が良い。
これには学習時のバッチサイズを100から25へ減らした影響も考えられる。
EmbeddingGemma でバッチサイズを25に落として追試すれば比較・検討できる。

ただ学習の取り回しの良さ、
および学習の効果の傾向が大きくは変わらない
という観点からは EmbeddingGemma のほうがより扱いやすい。

## EmbeddingGemmaをバッチサイズ25で学習

前節の検討・考察を受けてバッチサイズを減らしたことの影響を見るための追試。

全体的にバッチサイズが100だった時よりもわずか (0.01～0.02ポイント) Accuracy が下がっている。
QWen3は 0.05 ~ 0.10 ポイント低下していたので、バッチサイズの影響はそれに比べて 1/5 程度と、大きくない。

```
$ ./learning_experiment.sh

$ grep -nr accuracy: EmbeddingGemma/*.txt
EmbeddingGemma/accuracy-trained-000.txt:accuracy: 0.12773722627737227
EmbeddingGemma/accuracy-trained-020.txt:accuracy: 0.3607924921793535
EmbeddingGemma/accuracy-trained-040.txt:accuracy: 0.5813347236704901
EmbeddingGemma/accuracy-trained-060.txt:accuracy: 0.7773722627737226
EmbeddingGemma/accuracy-trained-080.txt:accuracy: 0.8957247132429614
EmbeddingGemma/accuracy-trained-100.txt:accuracy: 0.9457768508863399
EmbeddingGemma/result-castle-000.txt:accuracy: 0.08333333333333333
EmbeddingGemma/result-castle-100.txt:accuracy: 0.75
EmbeddingGemma/result-jreki-000.txt:accuracy: 0.04351830531890399
EmbeddingGemma/result-jreki-100.txt:accuracy: 0.2816025788625374
EmbeddingGemma/result-jrstation-000.txt:accuracy: 0.04167626064932074
EmbeddingGemma/result-jrstation-100.txt:accuracy: 0.27814874510706883
EmbeddingGemma/result-oldcountry-000.txt:accuracy: 0.03529411764705882
EmbeddingGemma/result-oldcountry-100.txt:accuracy: 0.611764705882353
```

*   [実験用スクリプト](./learning_experiment.sh)
