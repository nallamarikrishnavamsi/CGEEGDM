"""
Audit HMS data splits for patient/EEG leakage.
Usage:
    python src/audit_splits.py --data_root /home/dsamantaai/krishna/data
"""
import os, sys, argparse
import pandas as pd

def audit(data_root):
    splits = {
        'full106k_train' : 'full106k_train.csv',
        'full106k_val'   : 'full106k_val.csv',
        'full106k_test'  : 'full106k_test.csv',
    }

    dfs = {}
    for name, fname in splits.items():
        path = os.path.join(data_root, fname)
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
            print(f"{name:<20}: {len(dfs[name]):>6} rows")
        else:
            print(f"{name:<20}: NOT FOUND")

    print("\n" + "="*60)
    print("LEAKAGE AUDIT")
    print("="*60)

    split_names = list(dfs.keys())
    for i in range(len(split_names)):
        for j in range(i+1, len(split_names)):
            n1, n2 = split_names[i], split_names[j]
            df1, df2 = dfs[n1], dfs[n2]

            # EEG ID leakage
            eeg1 = set(df1['eeg_id'].astype(int))
            eeg2 = set(df2['eeg_id'].astype(int))
            eeg_overlap = eeg1 & eeg2

            # Patient ID leakage
            pat_overlap = set()
            if 'patient_id' in df1.columns and 'patient_id' in df2.columns:
                pat1 = set(df1['patient_id'].astype(int))
                pat2 = set(df2['patient_id'].astype(int))
                pat_overlap = pat1 & pat2

            status_eeg = "✅ CLEAN" if len(eeg_overlap) == 0 else f"❌ LEAK ({len(eeg_overlap)} EEGs)"
            status_pat = "✅ CLEAN" if len(pat_overlap) == 0 else f"❌ LEAK ({len(pat_overlap)} patients)"

            print(f"\n{n1} vs {n2}:")
            print(f"  EEG ID overlap    : {status_eeg}")
            print(f"  Patient overlap   : {status_pat}")

            if len(eeg_overlap) > 0:
                print(f"  Overlapping EEGs  : {list(eeg_overlap)[:5]}...")
            if len(pat_overlap) > 0:
                print(f"  Overlapping pats  : {list(pat_overlap)[:5]}...")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Critical check: pretrain vs finetune_test
    if 'full106k_train' in dfs and 'full106k_test' in dfs:
        tr_pat = set(dfs['full106k_train']['patient_id'].astype(int)) \
                 if 'patient_id' in dfs['full106k_train'].columns else set()
        te_pat = set(dfs['full106k_test']['patient_id'].astype(int)) \
                 if 'patient_id' in dfs['full106k_test'].columns else set()
        overlap = tr_pat & te_pat
        if len(overlap) == 0:
            print("✅ CRITICAL: No patient overlap between full106k_train and full106k_test")
        else:
            print(f"❌ CRITICAL: {len(overlap)} patients appear in both full106k_train and full106k_test!")
            print(f"   This invalidates the evaluation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        default='/home/dsamantaai/krishna/data')
    args = parser.parse_args()
    audit(args.data_root)


if __name__ == '__main__':
    main()
