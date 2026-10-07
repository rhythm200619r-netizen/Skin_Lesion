import pandas as pd
import os
import sys
sys.path.append('src')
from dataset import three_way_split, assert_no_patient_leakage
from evaluate import OUTPUTS_DIR

def merge_metadata(metadata_path, target_csv="data/train.csv"):
    if not os.path.exists(metadata_path):
        print(f"Metadata file {metadata_path} not found.")
        return
    df_meta = pd.read_csv(metadata_path)
    df_target = pd.read_csv(target_csv)
    
    # Keep only needed columns
    needed = ["image_name", "patient_id", "age_approx", "sex", "anatom_site_general_challenge"]
    df_meta = df_meta[[c for c in needed if c in df_meta.columns]]
    
    # Merge
    merged = df_target.merge(df_meta, on="image_name", how="left")
    merged.to_csv(target_csv, index=False)
    print(f"Merged metadata into {target_csv}. Columns are now: {list(merged.columns)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="data/ISIC_2020_Training_GroundTruth.csv")
    args = parser.parse_args()
    
    merge_metadata(args.metadata)
    
    # Now run split and print stats
    train_df = pd.read_csv("data/train.csv")
    if "patient_id" in train_df.columns:
        t_df, v_df, test_df = three_way_split(train_df)
        assert_no_patient_leakage(t_df, v_df, test_df)
        
        print(f"Train: {len(t_df)} images, {t_df['target'].sum()} malignant, {t_df['patient_id'].nunique()} patients")
        print(f"Val: {len(v_df)} images, {v_df['target'].sum()} malignant, {v_df['patient_id'].nunique()} patients")
        print(f"Test: {len(test_df)} images, {test_df['target'].sum()} malignant, {test_df['patient_id'].nunique()} patients")
        
        # Save splits
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        t_df.to_csv(os.path.join(OUTPUTS_DIR, "train_split.csv"), index=False)
        v_df.to_csv(os.path.join(OUTPUTS_DIR, "val_split.csv"), index=False)
        test_df.to_csv(os.path.join(OUTPUTS_DIR, "test_split.csv"), index=False)
        print("Saved split CSVs to outputs directory.")
    else:
        print("Cannot run real split: patient_id still missing.")
