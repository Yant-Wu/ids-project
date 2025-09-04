import os
import sys
import argparse
import pandas as pd


def find_csvs(root_dir: str) -> list[str]:
    csvs = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(r, f))
    csvs.sort()
    return csvs


def read_columns(file_path: str) -> list[str]:
    # 嘗試讀取標頭，必要時以 latin1 重試
    try:
        df = pd.read_csv(file_path, nrows=0, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, nrows=0, low_memory=False, encoding="latin1")
    return list(df.columns)


def stream_concat(csv_files: list[str], out_path: str, chunksize: int = 200_000) -> tuple[int, int]:
    # 預掃描所有欄位，統一 schema（欄位聯集）
    all_cols: set[str] = set()
    for fp in csv_files:
        cols = read_columns(fp)
        all_cols.update(cols)
    columns = sorted(all_cols)

    total_rows = 0
    wrote_header = False
    # 逐檔案逐分塊寫出，避免佔用大量記憶體
    for fp in csv_files:
        print(f"讀取: {fp}")
        try:
            itr = pd.read_csv(fp, chunksize=chunksize, low_memory=False)
        except UnicodeDecodeError:
            itr = pd.read_csv(fp, chunksize=chunksize, low_memory=False, encoding="latin1")
        for chunk in itr:
            total_rows += len(chunk)
            # 統一欄位順序與缺漏
            chunk = chunk.reindex(columns=columns)
            chunk.to_csv(out_path, mode="a" if wrote_header else "w", index=False, header=not wrote_header)
            wrote_header = True

    return total_rows, len(csv_files)


def main():
    parser = argparse.ArgumentParser(description="Merge CSVs under a folder (recursively) into one file.")
    parser.add_argument("--input", "-i", default="data", help="輸入資料夾（預設: data）")
    parser.add_argument("--output", "-o", default=os.path.join("data_final", "CICIDS2017.csv"), help="輸出檔案路徑（預設: data_final/CICIDS2017.csv）")
    parser.add_argument("--chunksize", type=int, default=200_000, help="讀取分塊大小（預設: 200000）")
    args = parser.parse_args()

    data_dir = args.input
    out_file = args.output
    out_dir = os.path.dirname(out_file) or "."

    # 檢查輸入資料夾是否存在
    if not os.path.isdir(data_dir):
        print(f"找不到輸入資料夾: {data_dir}")
        sys.exit(1)

    # 遞迴找出所有 csv 檔案
    files = find_csvs(data_dir)
    if not files:
        print(f"資料夾內沒有 CSV 檔: {data_dir}")
        sys.exit(1)

    # 確保輸出資料夾存在
    os.makedirs(out_dir, exist_ok=True)

    # 串流式合併與輸出
    total_rows, file_count = stream_concat(files, out_file, chunksize=args.chunksize)

    print(f"合併完成，讀取檔案數: {file_count}，總筆數：{total_rows}，輸出: {out_file}")


if __name__ == "__main__":
    main()
