import os
import glob
from beatmap import extract_osz

def list_osz_files(data_dir="data"): #data dir 내 파일명을 정렬하여 반환
    osz_files = sorted(glob.glob(os.path.join(data_dir, "*.osz")))
    if not osz_files:
        print("cannot find .osz files in the data directory.")
        return []

    print("Select songs to use.")
    for idx, path in enumerate(osz_files, 1):
        print(f" [{idx}] {os.path.basename(path)}")
    return osz_files


def select_song(data_dir="data"):
    osz_files = list_osz_files(data_dir)
    if not osz_files:
        return None

    try:
        choice = int(input("Select a song's number to choose(1-5): ")) - 1
    except ValueError:
        print("Write down a number.")
        return None

    if choice < 0 or choice >= len(osz_files):
        print("Out of range.")
        return None

    selected_osz = osz_files[choice] 
    print(f"\n Selected song: {os.path.basename(selected_osz)}")

    # osz 압축 해제
    out_dir = extract_osz(selected_osz)

    # 해당 폴더 내 .osu 파일 탐색
    osu_files = sorted(glob.glob(os.path.join(out_dir, "*.osu")))
    if not osu_files:
        print("There is no .osu file.")
        return None

    # 난이도 선택
    print("\n Select level:")
    for idx, path in enumerate(osu_files, 1):
        print(f" [{idx}] {os.path.basename(path)}")

    try:
        diff_choice = int(input("level number: ")) - 1
    except ValueError:
        diff_choice = 0  # 기본 첫 번째로 선택

    if diff_choice < 0 or diff_choice >= len(osu_files):
        diff_choice = 0

    selected_osu = osu_files[diff_choice]
    print(f"\n selected .osu file: {selected_osu}")
    return selected_osu
