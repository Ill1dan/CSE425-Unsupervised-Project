import os
import csv

PROCESSED_DIR = "data/processed"
OUT_CSV = "data/metadata.csv"

def main():
    rows = []
    idx = 0

    for lang in ["english", "bangla"]:
        lang_dir = os.path.join(PROCESSED_DIR, lang)
        for f in sorted(os.listdir(lang_dir)):
            if f.lower().endswith(".wav"):
                rows.append([
                    idx,
                    os.path.join(lang_dir, f),
                    lang,
                    "",
                    ""
                ])
                idx += 1

    os.makedirs("data", exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "filepath", "language", "title", "lyrics"])
        writer.writerows(rows)

    print("metadata.csv created")
    print("Total rows:", len(rows))


if __name__ == "__main__":
    main()
