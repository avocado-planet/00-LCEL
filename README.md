# LCEL（LangChain Expression Language）について

## LCELとは

LCEL（LangChain Expression Language）は、LangChainのコンポーネントを**パイプ演算子 `|`** で宣言的に繋ぐための構文です。2023年8月に導入され、旧来の `LLMChain` や `SequentialChain` を置き換える現在の推奨方式です。

### LCELの利点

- **簡潔な記述**: `prompt | model | parser` の1行でチェーンが完成
- **統一インターフェース**: `invoke`, `batch`, `stream`, `ainvoke` 等が自動的に使える
- **コンポーザブル**: チェーン同士を `|` で繋いで大きなパイプラインを構築できる
- **LangSmith連携**: トレーシングが自動的に有効になる

### 基本構造

```python
chain = prompt | model | parser
result = chain.invoke({"key": "value"})
```

各コンポーネントは `Runnable` インターフェースを実装しており、左の出力が右の入力になります。

---

## 全パターン一覧

### 1. 基本チェーン（Prompt → Model → Parser）

最も基本的なパターン。`|` で3つのコンポーネントを順次接続します。

```python
prompt = ChatPromptTemplate.from_template("{word}を説明してください。")
chain = prompt | model | parser
result = chain.invoke({"word": "LCEL"})
```

**データの流れ:**

```
{"word": "LCEL"} → Prompt → "LCELを説明してください。"
                 → Model  → AIMessage(content="LCELは...")
                 → Parser → "LCELは..."
```

`.pipe()` メソッドで書くこともできます（`|` と完全に等価）:

```python
chain = prompt.pipe(model).pipe(parser)
```

---

### 2. Runnableインターフェース（invoke / batch / stream / async）

LCELで作ったチェーンは、自動的に以下の呼び出しメソッドを持ちます。

| メソッド | 用途 | 非同期版 |
|---------|------|---------|
| `invoke(input)` | 1件処理 | `ainvoke()` |
| `batch([inputs])` | 複数件を並列処理 | `abatch()` |
| `stream(input)` | トークン単位の逐次出力 | `astream()` |

```python
# invoke: 1件
result = chain.invoke({"city": "東京"})

# batch: 複数件（max_concurrencyで同時実行数制御）
results = chain.batch(
    [{"city": "東京"}, {"city": "大阪"}],
    config={"max_concurrency": 2}
)

# stream: トークンごとに逐次出力
for chunk in chain.stream({"city": "札幌"}):
    print(chunk, end="", flush=True)

# async
result = await chain.ainvoke({"city": "名古屋"})
```

---

### 3. チェーンの連結（Chain → Chain）

チェーン自体も `Runnable` なので、チェーン同士を `|` で繋げます。
前のチェーンの出力（文字列）を次のチェーンの入力にマッピングする必要があります。

```python
# Step1: 分析
analyze = analyze_prompt | model | parser

# Step2: 要約（{text}を期待）
summarize = summarize_prompt | model | parser

# 連結：ラムダで出力を辞書に変換
full = analyze | (lambda text: {"text": text}) | summarize
```

**RunnablePassthroughを使う方法:**

```python
step2 = (
    {"analysis": RunnablePassthrough()}
    | ChatPromptTemplate.from_template("結論を抽出:\n{analysis}")
    | model | parser
)
full = step1 | step2
```

---

### 4. RunnableLambda（任意のPython関数をRunnable化）

Python関数をチェーンの一部にする方法は2つあります。

**方法1: 明示的なラップ**
```python
def word_count(text: str) -> str:
    return f"[{len(text.split())}語] {text}"

chain = prompt | model | parser | RunnableLambda(word_count)
```

**方法2: 暗黙変換（チェーン内で関数を `|` で繋ぐ）**
```python
def to_upper(text: str) -> str:
    return text.upper()

chain = prompt | model | parser | to_upper  # 自動的にRunnableLambdaに変換
```

**LLMなしの関数チェーン:**
```python
add_ten = RunnableLambda(lambda x: x + 10)
double  = RunnableLambda(lambda x: x * 2)
chain   = add_ten | double
chain.invoke(5)  # → 30
```

**辞書入力・辞書出力:**
```python
def enrich(data: dict) -> dict:
    data = data.copy()
    data["length"] = len(data["text"])
    return data

chain = RunnableLambda(enrich) | prompt | model | parser
```

---

### 5. RunnablePassthrough（入力をそのまま通す）

入力を変更せずにそのまま次に渡します。`RunnableParallel` と組み合わせて、元の入力を保持しつつ加工する場面で頻出します。

**基本:**
```python
passthrough = RunnablePassthrough()
passthrough.invoke("Hello")  # → "Hello"
```

**`.assign()` — 元の入力にキーを追加:**
```python
chain = RunnablePassthrough.assign(
    upper=lambda x: x["name"].upper(),
    length=lambda x: len(x["name"])
)
chain.invoke({"name": "langchain"})
# → {"name": "langchain", "upper": "LANGCHAIN", "length": 9}
```

**RAGパターン（最頻出）:**
```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | model | parser
)
rag_chain.invoke("Pythonとは？")
```

この場合、`"Pythonとは？"` が:
- `retriever` に渡されて文書を取得 → `context`
- `RunnablePassthrough()` でそのまま → `question`

---

### 6. RunnableParallel（並列実行と結果の合流）

複数のRunnableを同時に実行し、結果を辞書にまとめます。

```
          ┌─ chain_a ─→ result_a ─┐
input ──→│                        │──→ {"a": result_a, "b": result_b}
          └─ chain_b ─→ result_b ─┘
```

**明示的な記法:**
```python
parallel = RunnableParallel(
    pro=pro_chain,
    con=con_chain
)
result = parallel.invoke({"topic": "AI"})
# → {"pro": "...", "con": "..."}
```

**辞書リテラル記法（省略形）:**
```python
# チェーン内で辞書を書くと自動的にRunnableParallelになる
chain = (
    {"pro": pro_chain, "con": con_chain, "topic": itemgetter("topic")}
    | synthesis_prompt
    | model | parser
)
```

---

### 7. RunnableBranch（条件分岐・ルーティング）

入力に応じて異なるチェーンに振り分けます。`(条件関数, Runnable)` のペアを順番に評価し、最初に `True` になったものを実行します。最後の引数がデフォルト。

```python
router = RunnableBranch(
    (lambda x: "コード" in x["q"], tech_chain),
    (lambda x: "物語" in x["q"], creative_chain),
    general_chain,  # デフォルト
)
```

**カスタム関数による代替:**
```python
def custom_router(data: dict):
    if "コード" in data["q"]:
        return tech_chain.invoke(data)
    return general_chain.invoke(data)

router = RunnableLambda(custom_router)
```

---

### 8. itemgetter（辞書から特定キーを取り出す）

`operator.itemgetter` は辞書から指定キーの値を取り出すPython標準関数です。
`RunnableParallel` 内で元の入力の一部だけを次に渡す際に使います。

```python
from operator import itemgetter

chain = (
    {
        "summary": summary_chain,          # LLMで処理した結果
        "topic": itemgetter("topic"),       # 元の入力をそのまま
        "lang": itemgetter("lang"),         # 元の入力をそのまま
    }
    | final_prompt | model | parser
)
chain.invoke({"topic": "ML", "lang": "日本語"})
```

---

### 9. @chain デコレータ（関数をそのままRunnable化）

`@chain` デコレータを使うと、普通のPython関数を書く感覚でRunnableを定義できます。
内部で複数のチェーンを呼び出す複雑なロジックに適しています。

```python
from langchain_core.runnables import chain

@chain
def analyze_and_score(data: dict) -> dict:
    sentiment = sentiment_chain.invoke({"text": data["text"]})
    score = {"positive": 1.0, "negative": 0.0}.get(sentiment, 0.5)
    return {"sentiment": sentiment, "score": score}

# invoke, batch, stream が使える
result = analyze_and_score.invoke({"text": "最高！"})

# 他のRunnableと繋げられる
pipeline = analyze_and_score | format_chain
```

---

### 10. bind / configurable（モデルパラメータの動的設定）

**`bind()` — パラメータを固定:**
```python
model_with_stop = model.bind(stop=["\n"])
chain = prompt | model_with_stop | parser
```

**`configurable_fields()` — 実行時にパラメータを切り替え:**
```python
from langchain_core.runnables import ConfigurableField

configurable_model = model.configurable_fields(
    temperature=ConfigurableField(id="llm_temperature")
)

chain = prompt | configurable_model | parser

# 実行時に温度を変更
result = chain.with_config(
    configurable={"llm_temperature": 1.5}
).invoke({"topic": "AI"})
```

---

### 11. with_fallbacks / with_retry（エラーハンドリング）

**`with_fallbacks()` — フォールバック:**
```python
chain_with_fallback = (
    prompt | bad_model | parser
).with_fallbacks(
    [prompt | good_model | parser]
)
# bad_modelが失敗 → good_modelで自動再実行
```

**`with_retry()` — リトライ:**
```python
flaky = RunnableLambda(unreliable_func).with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=False
)
```

---

## パターン早見表

| パターン | 書き方 | ユースケース |
|---------|--------|-------------|
| パイプ | `a \| b \| c` | 順次処理 |
| .pipe() | `a.pipe(b)` | パイプの別記法 |
| invoke | `chain.invoke(x)` | 1件処理 |
| batch | `chain.batch([x, y])` | 複数件並列処理 |
| stream | `chain.stream(x)` | ストリーミング出力 |
| async | `await chain.ainvoke(x)` | 非同期実行 |
| RunnableLambda | `RunnableLambda(func)` | 任意関数をRunnable化 |
| 暗黙変換 | `chain \| func` | 関数の自動ラップ |
| RunnablePassthrough | `RunnablePassthrough()` | 入力をそのまま通す |
| .assign() | `RunnablePassthrough.assign(k=fn)` | 入力に新キーを追加 |
| RunnableParallel | `RunnableParallel(a=x, b=y)` | 並列実行 |
| 辞書リテラル | `{"a": chain_a}` | RunnableParallelの省略形 |
| RunnableBranch | `RunnableBranch((cond, run), default)` | 条件分岐 |
| itemgetter | `itemgetter("key")` | 辞書からキー取得 |
| @chain | `@chain def f(x): ...` | 関数をRunnable化 |
| bind | `model.bind(stop=[...])` | パラメータ固定 |
| configurable_fields | `model.configurable_fields(...)` | 実行時パラメータ切替 |
| with_fallbacks | `chain.with_fallbacks([alt])` | フォールバック |
| with_retry | `runnable.with_retry(...)` | リトライ |

---

## 参考: 元ファイルとの対応

| 元ファイルのセクション | 本ガイドでの対応 |
|-------------------|---------------|
| Prompt+Model+Parser連結 | セクション1 |
| stream / batch | セクション2 |
| chain連結 | セクション3 |
| 任意関数→RunnableLambda | セクション4 |
| RunnableParallel | セクション6 |
| itemgetter | セクション8 |
| *(なし)* | セクション5 (RunnablePassthrough), 7 (RunnableBranch), 9 (@chain), 10 (bind/configurable), 11 (fallbacks/retry), 12 (総合演習) |
