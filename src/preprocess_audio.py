import os
import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 22050
DURATION_SEC = 30
TARGET_LEN = TARGET_SR * DURATION_SEC

IN_DIR = "data/audio"
OUT_DIR = "data/processed"

ALLOWED_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def trim_or_pad(y):
    if len(y) > TARGET_LEN:
        return y[:TARGET_LEN]
    if len(y) < TARGET_LEN:
        return np.pad(y, (0, TARGET_LEN - len(y)))
    return y

def pick_middle_segment(y):
    n = len(y)
    if n <= TARGET_LEN:
        return trim_or_pad(y)

    mid = n // 2
    start = max(0, mid - TARGET_LEN // 2)
    end = start + TARGET_LEN

    if end > n:
        return y[-TARGET_LEN:]

    return y[start:end]


def process_file(in_path, out_path):
    try:
        y, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

        # Handle rare decode failures (empty audio)
        if y is None or len(y) == 0:
            return False

        y = pick_middle_segment(y)
        sf.write(out_path, y, TARGET_SR)
        return True

    except Exception:
        return False


def main():
    ensure_dir(OUT_DIR)

    processed = 0
    skipped = []
    total_seen = 0

    for lang in ["english", "bangla"]:
        in_lang = os.path.join(IN_DIR, lang)
        out_lang = os.path.join(OUT_DIR, lang)
        ensure_dir(out_lang)

        if not os.path.isdir(in_lang):
            print("[WARN] Missing folder:", in_lang)
            continue

        for f in sorted(os.listdir(in_lang)):
            if not f.lower().endswith(ALLOWED_EXTS):
                continue

            total_seen += 1
            in_path = os.path.join(in_lang, f)
            out_name = os.path.splitext(f)[0] + ".wav"
            out_path = os.path.join(out_lang, out_name)

            ok = process_file(in_path, out_path)
            if ok:
                processed += 1
                print("[OK]  ", lang + ":", f)
            else:
                skipped.append(in_path)
                print("[SKIP]", lang + ":", f)

    print("\n==== SUMMARY ====")
    print("Total audio files seen:", total_seen)
    print("Processed:", processed)
    print("Skipped:", len(skipped))

    if skipped:
        ensure_dir("results")
        log_path = os.path.join("results", "skipped_files.txt")
        with open(log_path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(skipped))
        print("Saved skipped list to:", log_path)

if __name__ == "__main__":
    main()
