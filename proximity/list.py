import csv

tsv_file = 'prefectures.tsv'  # TSVファイル名を指定
second_elements = []

with open(tsv_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        if len(row) > 1:
            second_elements.append(row[1])

print(second_elements)
