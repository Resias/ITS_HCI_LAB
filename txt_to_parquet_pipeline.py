
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TXT (pipe-delimited) -> Parquet pipeline
----------------------------------------
- Streams huge TXT files in chunks (no intermediate .npy)
- Applies a clean schema (keeps ID fields as strings to preserve leading zeros)
- Parses dates/datetimes
- Writes a single Parquet file or a partitioned Parquet dataset
- Compression: snappy (default), zstd, gzip supported
Usage:
    python txt_to_parquet_pipeline.py \
        --input /path/to/input.txt \
        --output /path/to/output_dir_or_file.parquet \
        --partition-by 운행일자 \
        --compression snappy \
        --chunksize 500000
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---- Column schema (ordered) ----
COLS = [
    "운행일자",
    "일련번호",
    "가상카드번호",
    "정산지역코드",
    "카드구분코드",
    "차량ID",
    "차량등록번호",
    "운행출발일시",
    "운행종료일시",
    "교통수단코드",
    "노선ID(국토부표준)",
    "노선ID(정산사업자)",
    "승차일시",
    "발권시간",
    "승차정류장ID(국토부표준)",
    "승차정류장ID(정산사업자)",
    "하차일시",
    "하차정류장ID(국토부표준)",
    "하차정류장ID(정산사업자)",
    "트랜잭션ID",
    "환승횟수",
    "사용자구분코드",
    "이용객수",
    "이용거리",
    "탑승시간",
]

# Columns that should remain string-like (IDs that may have leading zeros)
STRING_COLS = {
    "운행일자",  # yyyymmdd as string to avoid accidental int cast; we'll also produce a parsed date
    "일련번호",
    "가상카드번호",
    "정산지역코드",
    "카드구분코드",
    "차량ID",
    "차량등록번호",
    "교통수단코드",
    "노선ID(국토부표준)",
    "노선ID(정산사업자)",
    "승차정류장ID(국토부표준)",
    "승차정류장ID(정산사업자)",
    "하차정류장ID(국토부표준)",
    "하차정류장ID(정산사업자)",
    "트랜잭션ID",
    "사용자구분코드",
}

# Datetime columns in yyyymmddHHMMSS format
DATETIME_COLS = [
    "운행출발일시",
    "운행종료일시",
    "승차일시",
    "발권시간",
    "하차일시",
]

# Numeric metric columns
NUMERIC_COLS = [
    "환승횟수",
    "이용객수",
    "이용거리",
    "탑승시간",
]

def _build_dtype_map() -> Dict[str, str]:
    dtype = {}
    for c in COLS:
        if c in STRING_COLS:
            dtype[c] = "string"
        elif c in NUMERIC_COLS:
            dtype[c] = "Int64"  # pandas nullable int
        else:
            dtype[c] = "string"  # read as string first; we'll parse datetimes separately
    return dtype

def _try_read_csv(path: Path, chunksize: int, encoding_hint: Optional[str]) -> pd.io.parsers.TextFileReader:
    # Prefer utf-8; fall back to cp949 if needed
    encodings = [encoding_hint] if encoding_hint else ["utf-8", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            reader = pd.read_csv(
                path,
                sep="|",
                header=None,
                names=COLS,
                dtype=_build_dtype_map(),
                chunksize=chunksize,
                encoding=enc,
                na_values=["", " ", "NULL", "NaN", "nan"],
                keep_default_na=True,
                engine="python",
            )
            print(f"[info] Using encoding={enc}", file=sys.stderr)
            return reader
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to open file with tried encodings {encodings}: {last_err}")

def _parse_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATETIME_COLS:
        # Try common formats quickly; coerce invalid
        # Expecting yyyymmddHHMMSS (length 14) or yyyymmdd (length 8) in some logs
        df[col] = pd.to_datetime(df[col].str.strip(), errors="coerce", format="%Y%m%d%H%M%S")
        # If all NaT, try date-only
        if df[col].isna().all():
            df[col] = pd.to_datetime(df[col].str.strip(), errors="coerce", format="%Y%m%d")
    # 운행일자 (date) also as parsed date column
    if "운행일자" in df.columns:
        parsed = pd.to_datetime(df["운행일자"].str.strip(), errors="coerce", format="%Y%m%d")
        df["운행일자_date"] = parsed.dt.date
    return df

def _cast_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
    return df

def write_parquet_streaming(
    input_path: Path,
    output_path: Path,
    partition_by: Optional[List[str]] = None,
    compression: str = "snappy",
    chunksize: int = 500_000,
    encoding_hint: Optional[str] = None,
) -> None:
    """
    Stream-read the TXT and write to Parquet.
    If output_path is a directory or partition_by is provided, write a partitioned dataset.
    Otherwise write a single Parquet file.
    """
    reader = _try_read_csv(input_path, chunksize=chunksize, encoding_hint=encoding_hint)

    # Determine dataset mode
    dataset_mode = bool(partition_by) or output_path.suffix == "" or output_path.suffix == ".dir"
    if dataset_mode:
        output_path.mkdir(parents=True, exist_ok=True)

    writer = None
    schema = None
    row_count = 0

    try:
        for i, chunk in enumerate(reader):
            # Cleanup & typing
            chunk = _normalize_strings(chunk)
            chunk = _parse_datetime_cols(chunk)
            chunk = _cast_numeric_cols(chunk)

            # Arrow conversion
            table = pa.Table.from_pandas(chunk, preserve_index=False)

            # Initialize schema/writer if single-file mode
            if not dataset_mode and writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    where=str(output_path),
                    schema=schema,
                    compression=compression,
                    use_dictionary=True,
                )

            if dataset_mode:
                # Partitioned dataset write
                pq.write_to_dataset(
                    table=table,
                    root_path=str(output_path),
                    partition_cols=partition_by or [],
                    compression=compression,
                    use_dictionary=True,
                )
            else:
                # Append to single parquet file
                writer.write_table(table)

            row_count += table.num_rows
            if (i + 1) % 1 == 0:
                print(f"[info] Written {row_count:,} rows...", file=sys.stderr)

    finally:
        if writer is not None:
            writer.close()

    print(f"[done] Total rows written: {row_count:,}", file=sys.stderr)
    if dataset_mode:
        print(f"[done] Parquet dataset at: {output_path}", file=sys.stderr)
    else:
        print(f"[done] Parquet file at: {output_path}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Convert pipe-delimited TXT logs to Parquet")
    ap.add_argument("--input", required=True, type=Path, help="Input .txt file path")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output .parquet file path or directory (if partitioned)")
    ap.add_argument("--partition-by", nargs="*", default=None,
                    help="Column(s) to partition by (e.g., 운행일자_date or 교통수단코드)")
    ap.add_argument("--compression", default="snappy", choices=["snappy", "zstd", "gzip", "brotli"],
                    help="Parquet compression codec")
    ap.add_argument("--chunksize", type=int, default=500_000,
                    help="Rows per chunk for streaming read")
    ap.add_argument("--encoding", default=None, help="Optional encoding hint (utf-8/cp949/euc-kr)")
    args = ap.parse_args()

    write_parquet_streaming(
        input_path=args.input,
        output_path=args.output,
        partition_by=args.partition_by,
        compression=args.compression,
        chunksize=args.chunksize,
        encoding_hint=args.encoding,
    )

if __name__ == "__main__":
    main()
