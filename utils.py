import pandas as pd


def load_and_prepare(path: str):
    """讀取 CSV，標準化欄位名稱，輸出包含 text 與 label 的 DataFrame。

    - 自動移除欄位名稱前後空白（CICIDS 常見 ' Label'）。
    - 不分大小寫尋找 'label' 欄位並重命名為 'label'。
    - 將除 label 外的所有欄位串成 "col=value" 的文字作為 text。
    """
    # 嘗試不同編碼避免 UnicodeDecodeError
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin1")

    # 標準化欄位名稱：去除前後空白
    trimmed = {c: c.strip() for c in df.columns}
    df = df.rename(columns=trimmed)

    # 尋找 label 欄位（忽略大小寫）
    label_candidates = [c for c in df.columns if c.strip().lower() == "label"]
    if label_candidates:
        if "label" not in df.columns:
            # 將第一個候選欄位改名為 'label'
            df = df.rename(columns={label_candidates[0]: "label"})
    else:
        # 沒有 label 欄位時，嘗試常見替代名，否則報錯
        alt_candidates = [
            c for c in df.columns if c.strip().lower() in {"class", "attack", "category"}
        ]
        if alt_candidates:
            df = df.rename(columns={alt_candidates[0]: "label"})
        else:
            raise KeyError("無法在資料中找到標籤欄位（期望 'Label' 或 'label'）。")

    # 構建 text 欄位（排除 label）
    feature_cols = [c for c in df.columns if c != "label"]
    # 避免 NaN 干擾，將值轉為字串
    def row_to_text(row):
        pairs = []
        for col in feature_cols:
            val = row[col]
            pairs.append(f"{col}={val}")
        return " ".join(pairs)

    df["text"] = df.apply(row_to_text, axis=1)
    return df
