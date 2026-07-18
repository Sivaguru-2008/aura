import os
import sys
import json
import ast
import pandas as pd
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from schemas.clinical import FINDINGS, Finding

def check_keyword(text: str, keywords: list, negations: list = None) -> float:
    if negations is None:
        negations = ["no", "without", "free of", "clear of", "resolved", "negative for", "ruled out", "normal size", "not enlarged"]
    
    for kw in keywords:
        idx = text.find(kw)
        while idx != -1:
            preceding = text[max(0, idx - 35):idx]
            is_negated = False
            for neg in negations:
                if neg in preceding:
                    is_negated = True
                    break
            if not is_negated:
                return 1.0
            idx = text.find(kw, idx + 1)
    return 0.0

def label_report(report_text: str) -> dict:
    report_text = report_text.lower()
    
    labels = {}
    
    # Cardiomegaly
    labels[Finding.CARDIOMEGALY] = check_keyword(
        report_text, 
        ["cardiomegaly", "enlarged heart", "heart is enlarged", "cardiac silhouette is enlarged", "enlargement of the cardiac silhouette", "enlargement of the heart", "heart size is enlarged", "cardiomegaly is again seen", "cardiomegaly is seen"],
        ["no", "without", "normal", "stable", "unchanged", "resolved", "free of"]
    )
    
    # Pleural Effusion
    labels[Finding.EFFUSION] = check_keyword(
        report_text, 
        ["effusion", "pleural fluid", "fluid in the pleural", "pleural effusion", "pleural effusions"],
        ["no", "without", "clear", "resolved", "free of", "no new", "none"]
    )
    
    # Consolidation
    labels[Finding.CONSOLIDATION] = check_keyword(
        report_text, 
        ["consolidation", "consolidative", "confluent consolidation"],
        ["no", "without", "clear", "resolved", "free of", "none"]
    )
    
    # Pneumothorax
    labels[Finding.PNEUMOTHORAX] = check_keyword(
        report_text, 
        ["pneumothorax", "air in the pleural space", "collapsed lung"],
        ["no", "without", "clear", "resolved", "free of", "none"]
    )
    
    # Nodule / Mass
    labels[Finding.NODULE] = check_keyword(
        report_text, 
        ["nodule", "nodules", "mass", "masses", "lesion", "lesions", "pulmonary nodule"],
        ["no", "without", "clear", "resolved", "free of", "none", "no new"]
    )
    
    # Opacity
    labels[Finding.OPACITY] = check_keyword(
        report_text, 
        ["opacity", "opacities", "opacification", "infiltrate", "infiltrates", "consolidation", "focal consolidation"],
        ["no", "without", "clear", "resolved", "free of", "none", "no new"]
    )
    
    # Hyperinflation / COPD / Emphysema
    labels[Finding.HYPERINFLATION] = check_keyword(
        report_text, 
        ["hyperinflation", "hyperinflated", "emphysema", "copd", "overinflation"],
        ["no", "without", "clear", "resolved", "free of", "none"]
    )
    
    return labels

def create_manifest(dataset_dir: str, output_path: str):
    base_path = Path(dataset_dir)
    images_dir = base_path / "official_data_iccv_final"
    
    train_csv = base_path / "mimic_cxr_aug_train.csv"
    val_csv = base_path / "mimic_cxr_aug_validate.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Train/Val CSVs not found in {base_path}")
        
    print(f"Reading train dataset from {train_csv}...")
    df_train = pd.read_csv(train_csv)
    print(f"Reading val dataset from {val_csv}...")
    df_val = pd.read_csv(val_csv)
    
    # Concatenate both to parse combined
    df = pd.concat([df_train, df_val], ignore_index=True)
    print(f"Combined data shape: {df.shape}")
    
    rows = []
    missing_images = 0
    found_images = 0
    
    for idx, row in df.iterrows():
        try:
            # Parse images list and reports list
            img_list = ast.literal_eval(row["image"])
            report_list = ast.literal_eval(row["text"])
        except Exception as e:
            continue
            
        # Combine all reports for the subject and label
        combined_report = " ".join(report_list)
        labels_dict = label_report(combined_report)
        
        # Build labels vector matching FINDINGS order
        labels_vector = [labels_dict[fd] for fd in FINDINGS]
        
        # Match each image path
        for rel_img_path in img_list:
            # Path in CSV is e.g. "files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
            full_img_path = images_dir / rel_img_path
            
            if full_img_path.exists():
                rows.append({
                    "path": str(full_img_path.resolve()),
                    "labels": labels_vector
                })
                found_images += 1
            else:
                missing_images += 1
                
    print(f"Extraction complete: Found images = {found_images}, Missing images = {missing_images}")
    
    # Write to JSONL manifest
    with open(output_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            
    print(f"Manifest written to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="C:/Users/aruls/Desktop/aura/mimic_cxr_data", help="Directory of mimic_cxr_data")
    parser.add_argument("--output", default="C:/Users/aruls/Desktop/aura/manifest.jsonl", help="Output path for the manifest file")
    args = parser.parse_args()
    
    create_manifest(args.data_dir, args.output)
