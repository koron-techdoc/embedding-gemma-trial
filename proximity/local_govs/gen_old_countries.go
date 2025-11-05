package main

// Wikipedia: 国名 に記載された現都道府県名から旧国名への読み替えデータを、
// 旧国名から現都道府県名の1:nへ書き換えるプログラム。
// 元データは手動で成形してファイル内に直接埋め込んだ。
//
// 参照: https://ja.wikipedia.org/wiki/%E6%97%A7%E5%9B%BD%E5%90%8D
//
// 実行例: go run ./gen_old_countries.go > japan_old_countries.go

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

const new2old = `青森県	陸奥
岩手県	陸中
宮城県	陸前
秋田県	羽後	陸中
山形県	羽前	羽後
福島県	岩代	磐城
茨城県	常陸	下総
栃木県	下野
群馬県	上野
埼玉県	武蔵
千葉県	上総	下総	安房
東京都	武蔵
神奈川県	相模	武蔵
新潟県	越後	佐渡
富山県	越中
石川県	能登	加賀
福井県	越前	若狭
山梨県	甲斐
長野県	信濃
岐阜県	美濃	飛騨
静岡県	駿河	伊豆	遠江
愛知県	尾張	三河
三重県	伊勢	伊賀	志摩	紀伊
滋賀県	近江
京都府	山城	丹波	丹後
大阪府	摂津	和泉	河内
兵庫県	播磨	但馬	摂津	丹波	淡路
奈良県	大和
和歌山県	紀伊
鳥取県	因幡	伯耆
島根県	石見	出雲	隠岐
岡山県	備前	備中	美作
広島県	備後	安芸
山口県	周防	長門
徳島県	阿波
香川県	讃岐	備前
愛媛県	伊予
高知県	土佐
福岡県	筑前	筑後	豊前
佐賀県	肥前
長崎県	肥前	壱岐	対馬
熊本県	肥後
大分県	豊前	豊後
宮崎県	日向
鹿児島県	薩摩	大隅
沖縄県	琉球
`

var (
	old2new = map[string][]string{}
	oldList []string
)

func add(newName string, oldNames []string) {
	for _, old := range oldNames {
		newList, ok := old2new[old]
		if !ok {
			oldList = append(oldList, old)
		}
		old2new[old] = append(newList, newName)
	}
}

func run(w io.Writer, r io.Reader) error {
	cr := csv.NewReader(r)
	cr.Comma = '\t'
	for {
		cr.FieldsPerRecord = 0
		records, err := cr.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}
		add(records[0], records[1:])
	}

	for _, oldName := range oldList {
		newList := old2new[oldName]
		fmt.Fprintf(w, "%s\t%s\n", oldName, strings.Join(newList, "\t"))
	}

	return nil
}

func main() {
	err := run(os.Stdout, strings.NewReader(new2old))
	if err != nil {
		log.Fatal(err)
	}
}
