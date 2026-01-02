import os
import pandas as pd

RAW_DIR = r"C:\Users\bhand\ids-bias-project\data\Raw\CIC-IDS-2017"
OUT_DIR = os.path.join("data", "combined")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv") or ".pcap_ISCX" in f]

def load_csv_in_chunks(fname, chunksize=500_000):
    """Load CSV in chunks to avoid memory errors."""
    possible_paths = [
        os.path.join(RAW_DIR, fname),
        os.path.join(RAW_DIR, fname + ".csv"),
    ]
    for fpath in possible_paths:
        if os.path.exists(fpath):
            print(f"📂 Loading {os.path.basename(fpath)} in chunks ...")
            dfs = []
            try:
                for chunk in pd.read_csv(fpath, chunksize=chunksize, low_memory=False):
                    # Strip column spaces
                    chunk.columns = [c.strip() for c in chunk.columns]
                    # Add source file
                    chunk["Source_File"] = fname
                    dfs.append(chunk)
                combined = pd.concat(dfs, ignore_index=True)
                print(f"✅ Loaded {combined.shape[0]:,} rows, {combined.shape[1]} columns")
                return combined
            except Exception as e:
                print(f"❌ Error reading {fpath}: {e}")
                return None
    print(f"⚠️ File not found for {fname}, skipping...")
    return None


def main():
    print("🚀 Starting chunked combination of CIC-IDS-2017 CSVs...\n")
    combined_list = []

    for fname in FILES:
        df = load_csv_in_chunks(fname)
        if df is not None and not df.empty:
            combined_list.append(df)

    if not combined_list:
        print("❌ No valid data files loaded. Exiting.")
        return

    final_df = pd.concat(combined_list, ignore_index=True)
    out_path = os.path.join(OUT_DIR, "CICIDS2017_combined.csv")
    final_df.to_csv(out_path, index=False)
    print(f"\n💾 Saved combined dataset → {out_path}")

    if "Label" in final_df.columns:
        print("\n📊 Label distribution:")
        print(final_df["Label"].value_counts().head(10))


if __name__ == "__main__":
    main()
