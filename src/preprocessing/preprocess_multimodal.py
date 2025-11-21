#!/usr/bin/env python3
"""
preprocessing_multimodal.py

Preprocessing script that prepares data for multimodal bot detection.
- Creates train/test/val splits
- Preserves raw text (description) and image URLs
- Saves splits to cleaned folder

Usage:
python preprocessing_multimodal.py \
  --input data/raw/twitter_human_bots_dataset.csv \
  --output-dir data/cleaned \
  --test-size 0.2 \
  --seed 42
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_multimodal_data(input_csv: str, output_dir: str, 
                                test_size: float = 0.2, seed: int = 42):
    """
    Preprocess data for multimodal bot detection.
    
    The multimodal approach needs:
    - description (text for BERT)
    - profile_image_url (for ResNet/VGG)
    - account_type (label)
    
    We keep the raw columns and just do train/test/val splitting.
    """
    
    print(f"[*] Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Check required columns
    required_cols = ['description', 'profile_image_url', 'account_type']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Map labels to 0/1
    if df['account_type'].dtype == object:
        df['account_type'] = df['account_type'].map({'bot': 1, 'human': 0})
    
    print(f"\nClass distribution:")
    print(df['account_type'].value_counts())
    
    # Check for missing values in key columns
    print(f"\nMissing values:")
    print(f"  description: {df['description'].isna().sum()} ({df['description'].isna().sum()/len(df)*100:.1f}%)")
    print(f"  profile_image_url: {df['profile_image_url'].isna().sum()} ({df['profile_image_url'].isna().sum()/len(df)*100:.1f}%)")
    
    # Handle missing descriptions (fill with empty string, model will handle)
    df['description'] = df['description'].fillna('')
    
    # Handle missing image URLs (fill with placeholder, learned embedding will handle)
    # DO NOT DROP - the model has a learned missing_img_emb parameter for this!
    df['profile_image_url'] = df['profile_image_url'].fillna('MISSING_IMAGE')
    
    n_missing_images = (df['profile_image_url'] == 'MISSING_IMAGE').sum()
    if n_missing_images > 0:
        print(f"\n[*] Found {n_missing_images} rows with missing profile_image_url ({n_missing_images/len(df)*100:.1f}%)")
        print(f"    These will use learned missing-image embedding during training")
    
    # Split: 80% train, 20% test (from your original split)
    print(f"\n[*] Splitting data (test_size={test_size}, seed={seed})...")
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['account_type'],
        random_state=seed
    )
    
    # Further split test into val and test (50-50 as you requested)
    val_df, test_df = train_test_split(
        test_df,
        test_size=0.5,
        stratify=test_df['account_type'],
        random_state=seed
    )
    
    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\nClass distribution in each split:")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        counts = split_df['account_type'].value_counts()
        print(f"  {name}: Bot={counts.get(1, 0)}, Human={counts.get(0, 0)}")
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    # Save with important columns for multimodal training
    # Keep all columns in case you want to use metadata later
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n[+] Saved splits:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    
    # Save a combined file with split indicator (useful for some workflows)
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_path = os.path.join(output_dir, 'all_splits.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"  {combined_path} (with 'split' column)")
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data for multimodal bot detection"
    )
    parser.add_argument(
        '--input',
        default='data/raw/twitter_human_bots_dataset.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output-dir',
        default='data/cleaned',
        help='Output directory for processed splits'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for initial test split (before val/test split)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()
    
    train_df, val_df, test_df = preprocess_multimodal_data(
        args.input,
        args.output_dir,
        args.test_size,
        args.seed
    )
    
    print("\n[+] Preprocessing complete.")

if __name__ == '__main__':
    main()