# STEM Studio

BS RoFormerを使い、音源をドラッグ＆ドロップしてSTEM分離するMac向けデスクトップアプリです。

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
| アコギコード | **BTC large-vocabulary + MIDI補助** | アコギ単独音源を入力に推定します。統合欄と同じ168コード語彙を使います。 |
| 統合コード | **BTC large-vocabulary + MIDI補助** | 音声を主根拠に168コード＋不明・無和音を推定。持続ベースから転回形を付けます。 |

すべてローカルで推論します。各モデルの重みは初回のみダウンロードされ、分離モデルは `.cache/bs_roformer/`、Solititoは `.cache/solitito/` に保存されます。

デフォルトの分離経路は `BS RoFormer SW → Mega53 Acoustic Guitar` のカスケードです。SWでボーカル、ドラム、ベース、エレキギター、ピアノ、その他を分離し、その出力からMega53でアコギとギター残りを分離します。必要なWAVがすべて揃っていれば再処理せず、不足パートだけを処理します。BS RoFormerはApple SiliconではMLXを優先し、Torch/MPS、CPUの順にフォールバックします。

BPMと拍グリッドはBeat This!で推定し、`beat-grid.json`に保存します。`downbeats`が4/4の位相、`beats`が四分音符格子です。ドラムはADTOF-pytorch、ピアノはTranskun V2、ベースはBasic Pitch系の専用処理を使います。ベースの8分音符未満の装飾音はコード解析から除外し、ピアノは持続音と同時発音和音を中心に使います。

通常のコード解析はBTC large-vocabularyを主推定器に使います。`mix.wav` / `original.wav` があれば使用し、なければSWの6 stemを加算して入力します。`guitar.wav` とそのアコギ分割音源は二重加算しません。音源がないJSONのみのフォルダは従来のMIDI推定を利用します。

公式の約12MBの重みを `.cache/btc/` に保存し、リビジョンとSHA-256を固定して検証します。CPU・float32・最大4スレッドで、音源読み込みと推論は10秒単位です。BTCの対数確率に弱いピアノ・ベース・保存済みSolititoの証拠を加え、線形計算量のViterbiで整えます。四分音符への固定や小節内のコード種類数制限はなく、約93ms単位で変化を扱います。分数コードは各コード区間全体で優勢な持続ベースがある場合に付加します。

BTCの14品質は major / minor / dim / aug / m6 / 6 / m7 / mMaj7 / maj7 / 7 / dim7 / m7b5 / sus2 / sus4 です。add9・9・11・13はモデルの直接出力には含まれません。`X`は判定不能、`N.C.`は無和音です。実装は `analysis/btc_chords.py`、公式出典とライセンスは `vendor/btc/` にあります。正解譜との精度比較は未実施です。


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

**コード解析**ボタンで、アコギ欄と統合欄の両方をBTCで再推定します。アコギ欄は `acoustic-guitar.wav`、統合欄は全体ミックスを入力にします。MIDI・拍グリッドは保存済み結果を補助に使います。アコギ結果も `acoustic-guitar-chords.json` に上書きします。BTCの結果にはSolitito用の四分音符補完を適用しません。以前のBTC結果を再推定の根拠に循環利用することもありません。

既存の `chords.json`（手修正を含む）は自動更新しません。コードだけを新版で再推定する場合は、必要に応じて保存結果をバックアップし、対象フォルダを指定して以下を実行します。この操作はそのフォルダのコードの手修正も置き換えます。MIDIや拍グリッドは再推論しません。

```sh
.venv/bin/python -c 'from analysis.chord_estimator import load_or_estimate; load_or_estimate("outputs/曲名", force=True)'
```

コード推定の回帰テスト: `.venv/bin/python -m unittest discover -s tests -p 'test_*chord*.py' -v`。

`acoustic-guitar.wav` がある場合は、アコギ欄にもエレキ＋アコギのコードビューと同じ、コード名とダイアグラムが左へ流れるタイムラインを表示します。アコギ欄の **再コード解析** は、保存済み結果を使わず、[Solitito](https://github.com/greblus/solitito)の公式ONNXモデルで毎回アコギ音源を再解析します。解析後は統合コードも更新します。現在のコードと続く10秒のコードを再生位置に合わせて表示します。両方のコードビューには拍グリッドの小節頭（downbeats）を縦の破線で表示し、コードと同じ速度でスクロールします。小節頭が未推定の場合は小節線を表示しません。結果は参考用の `acoustic-guitar-chords.json` に保存され、フォルダを開き直した際に読み込まれます。従来のMIDI由来コードとは別に保持します。

初回のみ[公式モデルとDSP重み](https://huggingface.co/greblus/solitito-ai)（合計約32MB）を `.cache/solitito/` に取得し、以降はCPUでローカル推論します。音源のアップロードはありません。単音・無音判定は `N.C.`、モデルが区別しないsus2/sus4は `sus` と表示し、ダイアグラムの代わりに「sus2 / sus4 未判定」と表示します。分離時の混入や速いコード変化に影響されるため、正解譜ではなく比較用の推定結果です。モデルの文脈窓の中心に時刻を合わせ、短い判定の揺れを抑えています。

アコギ単独のBTC再解析は `.venv/bin/python -m analysis.btc_chords "outputs/曲名" --force` で実行できます。保存済みの旧Solitito結果は再解析するまで残り、画面に「Solitito（旧結果）」と表示します。以下のSolititoの補完仕様は旧結果にのみ適用します。

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
アコギ抽出は必ずカスケード経路で実行します。まずSWの6 stemを保存し、Mega53で `acoustic-guitar.wav` と `guitar-other.wav` を生成します。アコギ専用のSolitito表示は四分音符単位で、統合コードのBTC推定とは独立しています。
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

### 謝辞・外部モデル

本アプリは以下のオープンソース実装および学習済みモデルを利用しています。各ソフトウェア、モデル重み、データセットのライセンスと利用条件は、それぞれの配布元の表記に従ってください。

- [BS RoFormer Infer](https://github.com/openmirlab/bs-roformer-infer) — 6 stem分離の実装。使用する **BS RoFormer SW** 重みの配布条件にも従います。
- [BS-RoFormer-MVSep-Mega-53-stems](https://huggingface.co/noblebarkrr/BS-Roformer-MVSep-Mega-53-stems) — Mega53 Acoustic Guitarのチェックポイントと設定。
- [Beat This!](https://github.com/CPJKU/beat_this) — BPM、beat、downbeat推定。
- [Spotify Basic Pitch](https://github.com/spotify/basic-pitch) — ボーカルおよびベースの音高・ノート推定。
- [Transkun](https://github.com/linjialuo/Transkun) — ピアノMIDI推定。
- [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) — ドラムのイベント推定。
- [Solitito](https://github.com/greblus/solitito) と [Solitito AIモデル](https://huggingface.co/greblus/solitito-ai) — アコギコード推定。Solitito由来のDSP実装とライセンスは [vendor/solitito/LICENSE](vendor/solitito/LICENSE) に同梱しています。
