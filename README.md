# STEM Studio

## 解析アルゴリズムと使用モデル

### 使用モデル一覧

| 処理 | 使用モデル / 手法 | 出力と役割 |
| --- | --- | --- |
| 音源分離 | **BS RoFormer SW** | ボーカル、ドラム、ベース、ギター、ピアノ、その他の6 stemを分離します。 |
| アコギ分離 | **BS RoFormer Mega53 Acoustic Guitar** | SWのギター出力を入力に、アコギと残りのギター音を分離します。標準のカスケード経路です。 |
| BPM・拍・小節頭 | **Beat This!** (`final0`) | BPM、四分音符の `beats`、4/4の小節頭 `downbeats` を `beat-grid.json` に保存します。 |
| ボーカルMIDI | **Spotify Basic Pitch**（ICASSP 2022） | 音域をC3–C6へ絞り、単旋律化したノート列を保存します。 |
| ベースMIDI | **Spotify Basic Pitch**（ICASSP 2022） | E1–G4に絞り、優勢な最低音を単旋律化してルート推定に使います。 |
| ピアノMIDI | **Transkun V2** | ピアノロールとコード品質・テンション判定に使うノート列を生成します。 |
| ドラムMIDI | **ADTOF-pytorch** | kick / snare / tom / hi-hat / cymbal のonsetを生成します。利用できない環境ではスペクトルonset検出へフォールバックします。 |
| アコギコード | **Solitito v2** ONNX + DSP重み | アコギ音源から基本コード品質を推定します。出力品質は major, minor, maj7, 7, m7, m7♭5, dim7, aug, sus に限られます。 |

すべてローカルで推論します。各モデルの重みは初回のみダウンロードされ、分離モデルは `.cache/bs_roformer/`、Solititoは `.cache/solitito/` に保存されます。

デフォルトの分離経路は `BS RoFormer SW → Mega53 Acoustic Guitar` のカスケードです。SWでボーカル、ドラム、ベース、エレキギター、ピアノ、その他を分離し、その出力からMega53でアコギとギター残りを分離します。必要なWAVがすべて揃っていれば再処理せず、不足パートだけを処理します。BS RoFormerはApple SiliconではMLXを優先し、Torch/MPS、CPUの順にフォールバックします。

BPMと拍グリッドはBeat This!で推定し、`beat-grid.json`に保存します。`downbeats`が4/4の位相、`beats`が四分音符格子です。ドラムはADTOF-pytorch、ピアノはTranskun V2、ベースはBasic Pitch系の専用処理を使います。ベースの8分音符未満の装飾音はコード解析から除外し、ピアノは持続音と同時発音和音を中心に使います。

コード解析v8はアコギのSolitito推定を軸にしますが、タイミングはdownbeat、Beat This!のbeats、ドラムonsetで先に固定します。各四分音符セルで、Piano chromaをコード品質、Bassをroot/転回形として別々に採点します。境界scoreはBass変化`0.50`、Piano変化`0.35`、拍節`0.15`で、Drumは境界判定には使いません。

candidateと小節内の既出candidate集合を状態に持つbeam型Viterbiで系列を推定します。3種類目の追加に`-0.15`、4種類目以降に`-0.35`を新規追加時だけ適用し、2・4拍目の変更には`-0.50`のpriorを置きます。MIDI marginが`0.10`未満の曖昧セルだけ、Solitito confidenceに比例したbonusを現在セルと前後セルへ加えて2回目のViterbiを実行します。グリッドは動かさず、最後に同一candidateの連続区間だけを結合します。

コード解析の調整値は `analysis/chord_estimator.py` 冒頭の `PARAMS` に集約しています。

BS RoFormerを使い、音源をドラッグ＆ドロップしてSTEM分離するMac向けデスクトップアプリです。

## 起動

セットアップ済みなら、Finderで `STEM Studio.app` をダブルクリックします。
この `.app` はこのフォルダ内のPython環境を使うランチャーです。単独で移動せず、このフォルダ内で使用してください。
`start.command` をダブルクリックして起動することもできます。
ターミナルからは `.venv/bin/python app.py` でも起動できます。

1. 音源をドラッグ＆ドロップ（クリックでの選択も可能）。複数ファイルをまとめて選べます。
2. 必要なら保存先を変更し、モデルを選んで **分離する** をクリック。
3. 完了後、各パートを試聴するか **Finderで開く** からファイルを取り出します。

複数ファイルを選んだ場合は1曲ずつ順番に処理し、完了後は最後に処理した曲を画面に表示します。

すでに処理済みの結果を聴く場合は、**処理済みフォルダを開く** から `outputs/曲名/` を選ぶと、再推論せずにそのフォルダ内の `.wav` を試聴できます。

`drums.wav` がある結果フォルダでは、**ドラムMIDIを推定** ボタンからADTOF-pytorchで kick / snare / tom / hi-hat / cymbal のMIDI風イベントを推定できます。推定後は再生位置に合わせてドラムMIDIビューを表示します。推定結果は `drums-midi.json` として同じフォルダに保存され、次回以降は再利用されます。

`piano.wav` がある結果フォルダでは、**ピアノMIDIを推定** ボタンからTranskun V2でノートイベントを推定できます。推定後は再生位置に合わせてピアノロールを表示します。低音が上、高音が下です。推定結果は `piano-midi.json` と `piano-transkun.mid` として保存されます。

MIDIビュー内の **BPM/拍グリッドを推定** ボタンからBeat This!でbeats/downbeatsを推定できます。推定後は拍グリッドを表示し、ドラムMIDIビューはBeat This!のbeatに量子化して表示します。結果は `beat-grid.json` に保存されます。

再生速度は `0.5x / 0.75x / 1.0x / 1.25x / 1.5x` から選べます。

統合コード解析v8は、downbeatを4/4の絶対的な1拍目として位相を固定し、Beat This!のbeats/BPMから4分音符latticeを作ります。各拍±0.15拍のDrum MIDIを探し、kick/cymbalは強く、snare/tomは中程度、hi-hatは弱く一度だけ吸着して固定します。固定セルごとにPianoだけでコード品質、Bassだけでroot/inversionを採点し、境界スコア（Bass変化35%、Piano chroma変化30%、Drum onset 20%、拍節15%）を使ってMIDI-only Viterbiを実行します。MIDI marginが小さいセルだけ、Solitito confidenceに比例したbonusを現在位置1.0、前後1拍0.5で上位3候補へ加え、2回目のViterbiで系列を整えます。Bass MIDIの同音結合は行いません。最終処理は同一コードの結合だけです。

既存の `chords.json`（手修正を含む）は自動更新しません。コードだけを新版で再推定する場合は、必要に応じて保存結果をバックアップし、対象フォルダを指定して以下を実行します。この操作はそのフォルダのコードの手修正も置き換えます。MIDIや拍グリッドは再推論しません。

```sh
.venv/bin/python -c 'from chord_estimator import load_or_estimate; load_or_estimate("outputs/曲名", force=True)'
```

コード推定の回帰テスト: `.venv/bin/python -m unittest test_chord_estimator -v`。

`acoustic-guitar.wav` がある場合は、アコギ欄にもエレキ＋アコギのコードビューと同じ、コード名とダイアグラムが左へ流れるタイムラインを表示します。アコギ欄の **再コード解析** は、保存済み結果を使わず、[Solitito](https://github.com/greblus/solitito)の公式ONNXモデルで毎回アコギ音源を再解析します。解析後は統合コードも更新します。現在のコードと続く10秒のコードを再生位置に合わせて表示します。両方のコードビューには拍グリッドの小節頭（downbeats）を縦の破線で表示し、コードと同じ速度でスクロールします。小節頭が未推定の場合は小節線を表示しません。結果は参考用の `acoustic-guitar-chords.json` に保存され、フォルダを開き直した際に読み込まれます。従来のMIDI由来コードとは別に保持します。

初回のみ[公式モデルとDSP重み](https://huggingface.co/greblus/solitito-ai)（合計約32MB）を `.cache/solitito/` に取得し、以降はCPUでローカル推論します。音源のアップロードはありません。単音・無音判定は `N.C.`、モデルが区別しないsus2/sus4は `sus` と表示し、ダイアグラムの代わりに「sus2 / sus4 未判定」と表示します。分離時の混入や速いコード変化に影響されるため、正解譜ではなく比較用の推定結果です。モデルの文脈窓の中心に時刻を合わせ、短い判定の揺れを抑えています。

ターミナルからも `.venv/bin/python acoustic_chords.py "outputs/曲名"` で実行できます。**BPM/MIDI/アコギ解析**ボタンは、拍グリッド、各MIDI、アコギのSolitito推定を順番に実行し、保存済みファイルがあればスキップします。**コード解析**ボタンは、アコギを基準にした統合コード推定だけを毎回実行します。統合コードはアコギの境界とコード名を保持したまま、ピアノ・ベースの構成音とベース音、ドラムの発音位置で確度の低い箇所だけ補正し、エレキ＋アコギビューへ表示します。アコギ解析の `--force` はアコギだけを再推定したい場合に使えます。統合テスト: `.venv/bin/python -m unittest test_acoustic_chords -v`（モデル取得済みなら実モデルによる推論も検証）。

アコギのコードは4分音符（1拍）未満の区間を、前後のコードで補完します。拍グリッドのテンポ変化を反映し、拍がなければBPM、BPMもなければ暫定120 BPM（0.5秒）を使います。ちょうど4分音符のコードと `N.C.` は残します。前後が同じコードならつなぎ、異なる場合は元の区間の長さと推定信頼度を比べて、支持の強い側を延長します。同点は直前のコードを優先します。先頭・末尾は隣接するコードで補完します。無音判定や元からある空白はまたがず、候補がない区間は `N.C.` とします。補完した時間は信頼度を加点せず、統合区間の信頼度を時間加重で計算します。保存済みの結果にも読み込み時に適用され、再推論は不要です。元の区間を保持するため、後で拍グリッドを更新した場合も元データから判定し直します。

各パートのシークバーは、クリック・ドラッグで再生位置を選べます。現在位置と総時間も表示します。
再生前や一時停止中にも位置を指定でき、パートを切り替えてもそれぞれの位置を保持します。
シークバーを選択した状態で左右キーを押すと1秒ずつ移動します。

「パート別モデル」欄でモデルを選び、**分離する** をクリックします。

| 選択モデル | 保存するパート |
| --- | --- |
| BS RoFormer SW | ボーカル / ドラム / ベース / ギター / ピアノ / その他 |
| BS RoFormer SW → Mega53 Acoustic Guitar | SWの6パート全部 + アコギ / ギター残り |

平均合成は行いません。
BS RoFormer SWは6パートを1回の推論で保存します。
アコギ抽出は必ずカスケード経路で実行します。まずSWで `vocals.wav`、`drums.wav`、`bass.wav`、`guitar.wav`、`piano.wav`、`other.wav` を保存し、そのうえでMega53 Acoustic Guitarを使って `acoustic-guitar.wav` と `guitar-other.wav` も保存します。Solititoのフレーム出力は最初から4分音符セルごとに投票します。ベースは8分音符以上、ピアノは8分音符以上または短い同時和音を弱いジョイント事前分布として加えます。アコギモデルの出力を主投票にし、MIDIはN-best順位の補助に限定します。
複数モデルの結果を足しても元音源に厳密に戻るとは限りません（混入音や音量の違いがあり得ます）。

BS RoFormerはApple SiliconでMLX backendを優先します。MLXで動かない場合はTorch/MPS、それも使えない環境ではCPUに戻ります。
処理ログの `BS RoFormer active: ...` で実際に使われたbackend/deviceを確認できます。
音源は外部へ送信しません。
初回のみ各モデルのダウンロードにインターネット接続が必要です。
BS RoFormerの重みは `.cache/bs_roformer` に保存されます。
ダウンロード途中のモデルは完成した重みと区別し、キャンセル後も再取得できます。
Mega53 Acoustic GuitarはMac全体が固まりにくいよう、短めのチャンクとMLXメモリ上限付きで処理します。
処理時間は曲の長さとMacの性能によって変わります。
出力は24-bit WAV固定です。
標準の保存先は `outputs/曲名/` です。分離音はその直下に `vocals.wav` などの名前で保存されます。
同じ曲の必要な分離音がすでに揃っている場合は、再推論せずに既存結果を表示します。
キャンセル時には途中の出力が残る場合があります。不要な出力はFinderから削除できます。

## 初回セットアップ

Python 3.11、uv、FFmpegが必要です。Homebrewを利用する場合:

```sh
brew install uv ffmpeg
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
chmod +x start.command
./start.command
```

PyTorch / torchaudioは、Demucsの音声保存APIとの互換性のため2.5.1に固定しています。
失敗時にはアプリ内の処理ログを確認してください。モデル取得に失敗した場合は接続を確認して再実行できます。

中間はfloat32で計算し、24-bit出力時はクリップ防止のため必要に応じて両パートを同じ倍率で下げます。

検証: `.venv/bin/python test_playback.py`。

追加モデル検証: `.venv/bin/python test_specialists.py`（初回は公式モデルを取得）。

ダイアグラムのフォームは `chord_shape.json` に集約しています（`frets` は6弦→1弦、`-1` はミュート、`barre` は `[フレット, 開始弦index, 終了弦index]`、弦indexは0〜5）。旧 `chord_shapes.json` の手修正も移行済みです。コード表の編集・保存もこのファイルを使用します。フォームの自動生成・移調による補完は行わず、未登録のコードは「フォーム未登録」と表示します。バレーもJSONに指定したものを使います。

アコギのコード境界は、短い区間の補完後に拍位置（4分音符）へ揃えます。8分音符以下の細かい境界は表示しません。1小節単位に固定せず、テンポ変化に追従します。拍がなければBPM（未取得時は暫定120）を使用します。音源の先頭・末尾はそのまま保ち、同じ時刻に揃った境界から長さ0の区間ができた場合は除去します。保存済みの結果にも再推論なしで適用します。アコギビューのコードカードをクリックすると、エレキ＋アコギビューと同じ修正・挿入ダイアログを開けます。編集結果は `acoustic-guitar-chords.json` と `_raw_chords` に保存し、次回のグリッド更新でも保持します。
