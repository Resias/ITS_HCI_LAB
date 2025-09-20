import os
import pandas as pd
import json
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Optional
from glob import glob
from datetime import datetime


def get_latest_column_definition():
    # Find the latest column definition file
    column_files = glob('COMMONCD/ColumnDefinition_*.xlsx')
    if not column_files:
        raise FileNotFoundError("No column definition files found")
    latest_file = max(column_files)
    return latest_file

def read_column_definitions(excel_file):
    # Read the TCD sheet from the Excel file
    try:
        df_columns = pd.read_excel(excel_file, sheet_name='TCD')
        # Assuming the column name and description are in the first two columns
        column_info = df_columns.iloc[:, :2]
        return column_info
    except Exception as e:
        print(f"Error reading column definitions: {e}")
        return None

def process_data_files():
    # Get column definitions
    column_def_file = get_latest_column_definition()
    column_info = read_column_definitions(column_def_file)
    
    if column_info is None:
        print("No column information found.")
        return
    
    # Process all txt files in DATA_txt folder
    data_files = glob('DATA_txt/*.txt')
    
    for file_path in data_files:
        try:
            # Read the data file using | as delimiter and fill missing values
            data = pd.read_csv(file_path, delimiter='|', header=None, na_values=[''], keep_default_na=True)
            # Fill missing values with NaN if there are missing columns
            expected_columns = range(len(column_info))
            missing_columns = set(expected_columns) - set(data.columns)
            for col in missing_columns:
                data[col] = np.nan
            
            # If the number of columns matches with column definitions
            if data.shape[1] == len(column_info):
                # Convert to numpy array
                data_array = data.to_numpy()
                
                # Create output filename
                output_file = os.path.splitext(file_path)[0].replace('DATA_txt', 'DATA_npy')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Save as npy file
                np.save(output_file, data_array)
                print(f"Successfully processed: {file_path}")
            else:
                print(f"Column count mismatch in file: {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def split_bus_train():
    cur_path = os.getcwd()
    data_path = os.path.join(cur_path, 'DATA_npy')
    npy_list = os.listdir(data_path)
    bus_path = os.path.join(cur_path, 'DATA_npy_bus')
    train_path = os.path.join(cur_path, 'DATA_npy_train')
    os.makedirs(bus_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    for npy in npy_list:
        cur_data_path  = os.path.join(data_path, npy)
        cur_data = np.load(cur_data_path, allow_pickle=True)
        train_mask = pd.isna(cur_data[:, 6])
        bus_mask = ~train_mask

        train_data = cur_data[train_mask]
        bus_data = cur_data[bus_mask]
        
        base_name = os.path.splitext(npy)[0]
        
        # 분리된 데이터 저장
        if len(train_data) > 0:
            train_file = os.path.join(train_path, f"{base_name}_train.npy")
            np.save(train_file, train_data)
        
        if len(bus_data) > 0:
            bus_file = os.path.join(bus_path, f"{base_name}_bus.npy")
            np.save(bus_file, bus_data)

def selected_column(selected_columns):

    # 처리할 디렉토리와 저장할 디렉토리 설정
    source_dirs = ['DATA_npy_train', 'DATA_npy_bus']
    output_dirs = ['DATA_npy_train_parsing', 'DATA_npy_bus_parsing']

    # 각 디렉토리에 대해 처리
    for source_dir, output_dir in zip(source_dirs, output_dirs):
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 소스 디렉토리의 모든 .npy 파일 처리
        source_path = os.path.join(os.getcwd(), source_dir)
        npy_files = [f for f in os.listdir(source_path) if f.endswith('.npy')]
        
        for npy_file in npy_files:
            # 파일 로드
            file_path = os.path.join(source_path, npy_file)
            data = np.load(file_path, allow_pickle=True)
            
            # 선택된 열만 추출
            selected_data = data[:, selected_columns]
            
            # 새 파일로 저장
            output_file = os.path.join(output_dir, npy_file)
            np.save(output_file, selected_data)
            
            print(f"{source_dir}/{npy_file} 처리 완료")
            print(f"- 원본 shape: {data.shape}")
            print(f"- 처리후 shape: {selected_data.shape}")

def pars2column():
    # Excel 파일 경로
    excel_path = os.path.join(os.getcwd(), 'COMMONCD', 'ColumnDefinition_20220721.xlsx')

    # TCD 시트 읽기
    tcd_columns = pd.read_excel(excel_path, sheet_name='TCD')

    # 컬럼 정보를 담을 딕셔너리 생성
    columns_dict = {
        'index_to_name': {},  # 인덱스로 컬럼명 찾기
        'name_to_index': {},  # 컬럼명으로 인덱스 찾기
        'columns_info': []    # 전체 컬럼 정보
    }

    # 데이터 구성
    for idx, row in tcd_columns.iterrows():
        column_name = row['컬럼명']
        # 인덱스-컬럼명 매핑
        columns_dict['index_to_name'][str(idx)] = column_name
        columns_dict['name_to_index'][column_name] = idx
        # 전체 컬럼 정보
        columns_dict['columns_info'].append({
            'index': idx,
            'name': column_name,
            'description': row['설명'] if '설명' in row else ''
        })

    # JSON 파일로 저장
    output_json_path = os.path.join(os.getcwd(), 'column_info.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(columns_dict, f, ensure_ascii=False, indent=2)

    print(f"컬럼 정보가 {output_json_path}에 저장되었습니다.")

    # 저장된 내용 확인
    print("\n저장된 컬럼 정보:")
    print('-' * 50)
    with open(output_json_path, 'r', encoding='utf-8') as f:
        loaded_columns = json.load(f)
        print("사용 예시:")
        print(f"1. 인덱스로 컬럼명 찾기: columns['index_to_name']['2'] = {loaded_columns['index_to_name']['2']}")
        print(f"2. 컬럼명으로 인덱스 찾기: columns['name_to_index']['{list(loaded_columns['name_to_index'].keys())[0]}'] = {loaded_columns['name_to_index'][list(loaded_columns['name_to_index'].keys())[0]]}")
        print("\n전체 컬럼 정보 중 첫 번째 항목:")
        print(json.dumps(loaded_columns['columns_info'][0], ensure_ascii=False, indent=2))


def to_np_datetime(val):
    # 결측 처리
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return np.datetime64('NaT')

    # 이미 datetime류면 바로 변환
    if isinstance(val, (np.datetime64, pd.Timestamp, datetime)):
        return np.datetime64(val)

    # 문자열/숫자 뒤섞임(.0 포함 등) → 숫자만 추출하여 포맷 판단
    s = re.sub(r'\D', '', str(val))
    if len(s) >= 14:  # YYYYMMDDHHMMSS
        s = s[:14]
        try:
            return np.datetime64(datetime.strptime(s, "%Y%m%d%H%M%S"))
        except Exception:
            return np.datetime64('NaT')
    elif len(s) == 8:  # YYYYMMDD
        try:
            return np.datetime64(datetime.strptime(s, "%Y%m%d"))
        except Exception:
            return np.datetime64('NaT')
    else:
        return np.datetime64('NaT')


def final_parsing(transport_dict, subway_df, max_rows_per_file=100_000):
    source_dirs = ['DATA_npy_train_parsing', 'DATA_npy_bus_parsing']
    output_dirs = ['train_pars_final', 'bus_pars_final']

    # 각 디렉토리에 대해 처리
    for source_dir, output_dir in zip(source_dirs, output_dirs):
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 소스 디렉토리의 모든 .npy 파일 처리
        source_path = os.path.join(os.getcwd(), source_dir)
        npy_files = [f for f in os.listdir(source_path) if f.endswith('.npy')]
        
        for npy_file in npy_files:
            # 파일 로드
            file_path = os.path.join(source_path, npy_file)
            data = np.load(file_path, allow_pickle=True)
            base_name = os.path.splitext(npy_file)[0]
            part_idx = 0
            buffer = []

            # --- 이미 처리된 파일이 있는지 확인 ---
            existing_parts = [f for f in os.listdir(output_dir) if f.startswith(base_name + "_final_part")]
            if existing_parts:
                print(f"스킵: {npy_file} (이미 처리됨, {len(existing_parts)}개 파트 존재)")
                continue
            # -----------------------------------
            empty_subway = 0
            for di in data:
                transport_code = di[1]
                transport_name = transport_dict.get(transport_code, '알 수 없음')
                
                date1 = to_np_datetime(di[2])
                date2 = to_np_datetime(di[5])

                if np.isnan(di[4]):
                    subway1 = None
                elif subway_df[subway_df["역번호"] == int(di[4])].empty:
                    empty_subway += 1
                    subway1 = None
                else:
                    subway1 = subway_df[subway_df["역번호"] == int(di[4])]["노드역명"].values[0]


                if np.isnan(di[7]):
                    subway2 = None
                elif subway_df[subway_df["역번호"] == int(di[7])].empty:
                    empty_subway += 1
                    subway2 = None
                else:
                    subway2 = subway_df[subway_df["역번호"] == int(di[7])]["노드역명"].values[0]

                try:
                    converted_row = np.array([
                        str(di[0]),
                        (transport_code, transport_name),
                        np.datetime64(date1),
                        (di[3], di[4], subway1),
                        np.datetime64(date2),
                        (di[6], di[7], subway2),
                        int(di[8]),
                        int(di[9])],
                        dtype=object)
                except Exception as e:
                    print(f"Error processing row in file {npy_file}: {e}")
                    print(subway_df[subway_df["역번호"] == di[4]]["노드역명"], subway_df[subway_df["역번호"] == di[7]]["노드역명"])
                    print(di)
                buffer.append(converted_row)

                # 버퍼가 임계치에 도달하면 파트로 저장
                if len(buffer) >= max_rows_per_file:
                    part_array = np.array(buffer, dtype=object)
                    out_path = os.path.join(output_dir, f"{base_name}_final_part{part_idx:03d}.npy")
                    np.save(out_path, part_array)
                    print(f"저장: {out_path} (rows={len(part_array)})")
                    buffer.clear()
                    part_idx += 1
            print(f"오류가 발생한 {empty_subway}개의 역번호가 존재합니다.")
            # 남은 버퍼 저장
            if buffer:
                part_array = np.array(buffer, dtype=object)
                out_path = os.path.join(output_dir, f"{base_name}_final_part{part_idx:03d}.npy")
                np.save(out_path, part_array)
                print(f"저장: {out_path} (rows={len(part_array)})")
            
            print(f"{source_dir}/{npy_file} 처리 완료")
            print(f"- 원본 shape: {data.shape}")
            print(f"- 생성된 파일 수: {part_idx + (1 if len(buffer)==0 else 0)}")



SEP_REGEX = re.compile(r"\s{2,}")

def _read_text(path: str) -> List[str]:
    p = Path(path)
    for enc in ("utf-8-sig", "cp949"):
        try:
            return p.read_text(encoding=enc).splitlines()
        except UnicodeDecodeError:
            continue
    # 마지막 시도
    return p.read_text(encoding="utf-8", errors="replace").splitlines()

def _is_rule(line: str) -> bool:
    s = line.strip()
    return len(s) > 0 and set(s) <= {"-", "─", "—", "━"}

def _split_fields(line: str) -> List[str]:
    # 좌우 공백 제거 후 2칸 이상 공백 기준 split
    return [tok.strip() for tok in SEP_REGEX.split(line.strip()) if tok.strip() != ""]

def _clean_header(name: str) -> str:
    # 괄호/물음표 등 제거, 양끝 공백 제거
    name = re.sub(r"[()?\uFF08\uFF09]", "", name)  # 전각 괄호도 제거
    name = re.sub(r"\s+", " ", name).strip()
    return name

def _coerce_bool(x: str) -> Optional[bool]:
    """
    표에 자주 있는 'X', 'O', 'Y/N', '예/아니오', 공백을 불리언으로 표준화.
    """
    if x is None:
        return np.nan
    s = str(x).strip().upper()
    if s in {"O", "Y", "YES", "TRUE", "T", "1"}:
        return True
    if s in {"X", "N", "NO", "FALSE", "F", "0"}:
        return False
    if s == "":
        return np.nan
    return np.nan

def parse_transport_txt(path: str) -> pd.DataFrame:
    lines = _read_text(path)
    print(f"Parsing {path} with {len(lines)} lines...")
    # 1) 유효 라인만 추출 (구분선/빈 줄 제거)
    payload = [ln for ln in lines if (ln.strip() != "" and not _is_rule(ln))]
    if not payload:
        raise ValueError("입력 파일에 유효한 데이터가 없습니다.")

    # 2) 헤더 파싱
    header_raw = payload[0]
    headers = [_clean_header(h) for h in _split_fields(header_raw)]
    data_rows = payload[1:]

    # 3) 데이터 행 파싱
    rows: List[List[str]] = []
    for ln in data_rows:
        cols = _split_fields(ln)

        # 컬럼 개수 불일치 시 보정 (부족하면 채우고, 많으면 자름)
        if len(cols) < len(headers):
            cols += [""] * (len(headers) - len(cols))
        elif len(cols) > len(headers):
            # 마지막 컬럼에 초과 토큰 합치기 (역명 등에 공백 많을 때 대비)
            head = cols[:len(headers)-1]
            tail_joined = " ".join(cols[len(headers)-1:])
            cols = head + [tail_joined]

        rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)

    # 4) 컬럼별 타입 변환 (상황에 맞게 조정)
    # 숫자형 후보
    int_candidates = ["교통운영", "교통수단코드", "호선코드", "역번호"]
    for c in int_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False), errors="coerce").astype("Int64")

    # 불리언 후보 (질문에 있는 3개)
    bool_candidates = [
        "민자포함", 
        "공항철도독립요금역사", 
        "수도권경계외요금",
    ]
    for c in bool_candidates:
        # 헤더 정리가 약간 다를 수 있어 contains로 보강
        match_cols = [col for col in df.columns if c in col]
        for mc in match_cols:
            df[mc] = df[mc].apply(_coerce_bool)

    # 날짜/시간 컬럼이 있다면 여기서 변환 규칙 추가 (예: 'YYYYMMDDHHMMSS' → datetime)
    # 예시:
    # if "기준시각" in df.columns:
    #     df["기준시각"] = pd.to_datetime(df["기준시각"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")

    return df



if __name__ == "__main__":
    # 1.
    process_data_files()
    # 2.
    split_bus_train()
    # 3.
    pars2column()
    
    # 4.
    output_json_path = os.path.join(os.getcwd(), 'column_info.json')
    with open(output_json_path, 'r', encoding='utf-8') as f:
        loaded_columns = json.load(f)
    for i in loaded_columns['columns_info']:
        print(i['index'], i['name'])
    
    # 5.
    all_columns = np.array([
        '운행일자','일련번호','가상카드번호','정산지역코드','카드구분코드','차량ID',\
        '차량등록번호','운행출발일시','운행종료일시','교통수단코드',\
        '노선ID(국토부표준)','노선ID(정산사업자)','승차일시','발권시간',\
        '승차정류장ID(국토부표준)','승차정류장ID(정산사업자)','하차일시',\
        '하차정류장ID(국토부표준)','하차정류장ID(정산사업자)','트랜잭션ID',\
            '환승횟수','사용자구분코드','이용객수','이용거리','탑승시간'])
    # 가상카드번호, 교통수단코드, 승차일시, 승차정류장ID(국토부표준),
    # 승차정류장ID(정산사업자), 하차일시, 하차정류장ID(국토부표준), 하차정류장ID(정산사업자),
    # 트랜잭션ID, 환승횟수
    selected_columns_idx = [2, 9, 12, 14, 15, 16, 17, 18, 19, 20]
    selected_column(selected_columns_idx)

    # 6.
    cd_tfcmn_path = os.path.join(os.getcwd(), 'COMMONCD', 'CD_TFCMN.txt')
    tfcmn_df = pd.read_csv(cd_tfcmn_path, sep='|', header=None)
    transport_dict = dict(zip(tfcmn_df.iloc[:, 1], tfcmn_df.iloc[:, 2]))
    subway_df = parse_transport_txt("COMMONCD/2022_수도권지하철_역사.txt")
    final_parsing(transport_dict, subway_df)
    print("Parser")