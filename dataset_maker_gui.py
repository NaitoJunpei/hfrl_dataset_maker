### jsonl形式のデータセットを読み込み、いずれかを選択し、
### 選択情報をjsonl形式で出力するGUIアプリケーション
### 読み込み用のjsonlファイルは、text1、text2の2属性、または、それに加えてindex属性を持つ
### 読み込み用のjsonlファイルには、処理済みフラグを追加しておく
### 出力用のjsonlファイルは、chosenとrejected、prompt、indexの4属性を持つことにする

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import json
import os
import sys
import re

from datasets import load_dataset, load_from_disk
import huggingface_hub

# データセットの読み込み
def load_dataset_from_directory(jsonl_path):
    """
    dataset = load_dataset("json", data_files=jsonl_path, split='train')
    # 処理済みフラグがない場合は、処理済みフラグをFalseにする
    if 'processed' not in dataset.column_names :
        dataset = dataset.map(lambda example : {'index' : None, 'processed' : False, 'text1' : example['text1'], 'text2' : example['text2'], 'prompt' : "", 'chosen' : "", 'rejected' : ""})
    return dataset
    """
    dataset = []
    with open(jsonl_path, 'r', encoding="utf-8") as f :
        for line in f :
            example = json.loads(line)
            # 処理済みフラグがない場合は、処理済みフラグをFalseにする
            if 'processed' not in example :
                example['processed'] = False
                example['index'] = None
                example['prompt'] = ""
                example['chosen'] = ""
                example['rejected'] = ""

            dataset.append(example)
    return dataset

# データセットの保存
def save_dataset_to_jsonl(dataset, target_path):
    dataset.save_to_disk(target_path)

# テキストからのプロンプト抽出
# 一番最初の### 応答:までの部分がプロンプトである
def extract_prompt_from_text(text):
    prompt = text.split('### 応答:')[0]
    return prompt

# GUIの作成
# プロンプト表示用のテキストボックス、
# text1表示、及び編集用のテキストボックス、
# text2表示、及び編集用のテキストボックス、
# 好みのデータを選択、あるいは却下するためのラジオボタン
# 戻る、次へボタン
# で構成される

class DatasetMakerGUI(tk.Frame):
    def __init__(self, dataset_path, output_path, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('HFRL Dataset Maker')
        self.master.geometry('800x600')
        self.master.resizable(width=False, height=False)
        self.pack()

        # テキストボックスの配置
        # プロンプト表示用のテキストボックス
        self.prompt_flame = tk.Frame(self)
        self.prompt_label = tk.Label(self.prompt_flame, text="プロンプト")
        self.prompt_label.pack(side='top')
        self.prompt_textbox = tk.Text(self.prompt_flame, height=10, width=60)
        self.prompt_textbox.pack(side='top')
        self.prompt_flame.pack(side='top')

        # text1表示、及び編集用のテキストボックス
        self.texts_flame = tk.Frame(self)
        self.text1_flame = tk.Frame(self.texts_flame)
        self.text1_label = tk.Label(self.text1_flame, text="text1")
        self.text1_label.pack(side='top')
        self.text1_textbox = tk.Text(self.text1_flame, height=20, width=60)
        self.text1_textbox.pack(side='top')
        self.text1_flame.pack(side='left')
        # text2表示、及び編集用のテキストボックス
        self.text2_flame = tk.Frame(self.texts_flame)
        self.text2_label = tk.Label(self.text2_flame, text="text2")
        self.text2_label.pack(side='top')
        self.text2_textbox = tk.Text(self.text2_flame, height=20, width=60)
        self.text2_textbox.pack(side='top')
        self.text2_flame.pack(side='left')
        self.texts_flame.pack(side='top')

        # text1, text2のうち、好みのデータを選択、あるいは両方却下するためのボタン
        self.button_flame = tk.Frame(self)
        self.button_label = tk.Label(self.button_flame, text="文章選択")
        self.button_label.pack(side='top')        
        self.button_text1 = tk.Button(self.button_flame, text="左", command=self.chosen1)
        self.button_text1.pack(side='left')
        self.button_text2 = tk.Button(self.button_flame, text="右", command=self.chosen2)
        self.button_text2.pack(side='left')
        self.button_all_reject = tk.Button(self.button_flame, text="却下", command=self.reject)
        self.button_all_reject.pack(side='left')
        self.button_flame.pack(side='top')

        # 戻る、次へボタン
        self.navigation_flame = tk.Frame(self)
        self.button_back = tk.Button(self.navigation_flame, text="戻る", command=self.back)
        self.button_back.pack(side='left')
        self.button_next = tk.Button(self.navigation_flame, text="次へ", command=self.next)
        self.button_next.pack(side='left')
        self.navigation_flame.pack(side='top')
        self.button_quit = tk.Button(self, text="終了", command=self.master.destroy)

        # データセットの読み込み
        self.dataset = load_dataset_from_directory(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_length = len(self.dataset)
        # 最初の、処理済みフラグがFalseのデータのindexを取得する
        self.index = 0
        while self.dataset[self.index]['processed'] :
            self.index += 1
        self.show_data()

        # 出力用のjsonlファイルのパス
        self.output_path = output_path

    # データの表示
    def show_data(self) :
        # データセットの最後まで表示した場合
        if self.index == self.dataset_length :
            messagebox.showinfo('メッセージ', 'データセットの最後まで表示しました。')
            # 最初のデータを表示する
            self.index = 0
        # プロンプトの表示
        prompt = extract_prompt_from_text(self.dataset[self.index]['text1'])
        self.prompt_textbox.delete('1.0', 'end')
        self.prompt_textbox.insert('1.0', prompt)
        # text1の表示
        self.text1_textbox.delete('1.0', 'end')
        self.text1_textbox.insert('1.0', self.dataset[self.index]['text1'][len(prompt):])
        # text2の表示
        self.text2_textbox.delete('1.0', 'end')
        self.text2_textbox.insert('1.0', self.dataset[self.index]['text2'][len(prompt):])

    # データの保存
    # button_text1、button_text2のどちらかが押されたときに呼び出される共通処理
    def save_data(self, chosen_text_id) :
        text1 = self.prompt_textbox.get('1.0', 'end') + self.text1_textbox.get('1.0', 'end')
        text2 = self.prompt_textbox.get('1.0', 'end') + self.text2_textbox.get('1.0', 'end')
        prompt = extract_prompt_from_text(self.dataset[self.index]['text1'])
        chosen = self.dataset[self.index]['text1'] if chosen_text_id == 1 else self.dataset[self.index]['text2']
        rejected = self.dataset[self.index]['text2'] if chosen_text_id == 1 else self.dataset[self.index]['text1']
        # datasetのindex番目のデータを更新する
        self.dataset[self.index]['index'] = self.index
        self.dataset[self.index]['text1'] = text1
        self.dataset[self.index]['text2'] = text2
        self.dataset[self.index]['processed'] = True
        self.dataset[self.index]['prompt'] = prompt
        self.dataset[self.index]['chosen'] = chosen
        self.dataset[self.index]['rejected'] = rejected

        # 出力用のjsonlファイルに、選択されたデータを追記する
        self.save_output_data()
        # 次のデータを表示する
        self.index += 1
        self.show_data()

    # jsonlファイルに、データセットを保存する
    def save_output_data(self) :
        # 出力用のjsonlファイルに、datasetを記載する
        with open(self.output_path, 'w', encoding="utf-8") as f :
            for example in self.dataset :
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')


    # 次へボタンが押されたときの処理
    def next(self) :
        # 次のデータを表示する
        self.index += 1
        self.show_data()

    # 戻るボタンが押されたときの処理
    def back(self) :
        # 前のデータを表示する
        self.index -= 1
        self.show_data()

    # button_text1が押されたときの処理
    def chosen1(self) :
        self.save_data(1)

    # button_text2が押されたときの処理
    def chosen2(self) :
        self.save_data(2)

    # button_all_rejectが押されたときの処理
    def reject(self) :
        # datasetのindex番目のデータについて、処理済みフラグをTrueにする
        self.dataset[self.index]['processed'] = True
        self.save_output_data()
        # 次のデータを表示する
        self.index += 1
        self.show_data()

# データセットの読み込み用のjson (jsonl)ファイルを選択する
# 最後に選択したディレクトリを初期ディレクトリとする
def select_dataset_path() :
    initialdir = ''
    if os.path.exists('dataset_path.txt') :
        with open('dataset_path.txt', 'r', encoding="utf-8") as f :
            initialdir = f.read()
    else :
        initialdir = os.path.abspath(os.path.dirname(__file__))
    dataset_path = filedialog.askopenfilename(
        title = 'データセットの読み込み用のjsonlファイルの選択',
        filetypes = [('jsonファイル', '*.json'), ('jsonlファイル', '*.jsonl')],
        initialdir = initialdir
    )
    if dataset_path != '' :
        with open('dataset_path.txt', 'w', encoding="utf-8") as f :
            f.write(os.path.dirname(dataset_path))
    return dataset_path

# 出力用のjsonlファイルを選択する
def select_output_path() :
    output_path = filedialog.asksaveasfilename(
        title = '出力用のjsonlファイルの選択',
        filetypes = [('jsonlファイル', '*.jsonl')],
        initialdir = os.path.abspath(os.path.dirname(__file__))
    )
    return output_path

# main関数
def main() :
    # データセットの読み込み用のjsonlファイルを選択する
    dataset_path = select_dataset_path()
    print(f"dataset_path : {dataset_path}")
    if dataset_path == '' :
        sys.exit()
    # 出力用のjsonlファイルを選択する
    #output_path = select_output_path()
    output_path = dataset_path
    if output_path == '' :
        sys.exit()
    # GUIの作成
    root = tk.Tk()
    app = DatasetMakerGUI(dataset_path, output_path, master=root)
    app.mainloop()

if __name__ == '__main__' :
    main()