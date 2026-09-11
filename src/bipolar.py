"""
Bipolar TCP montage conversion for HMS EEG, matching the exact 22-channel
derivation order used by original EEGDM/TUEV preprocessing
(EEGDM_original/src/preprocessing.py: bipolar_ch_order).

HMS's raw parquet channels (verified against an actual sample file,
2026-09-02): Fp1, F3, C3, P3, F7, T3, T5, O1, Fz, Cz, Pz, Fp2, F4, C4,
P4, F8, T4, T6, O2, EKG. No A1/A2 (mastoid reference) electrodes exist
in this dataset.

Consequence: 20 of the 22 EEGDM bipolar derivations are constructible.
"A1-T3" (index 9) and "A2-T4" (index 14) CANNOT be constructed --
A1 and A2 are not recorded electrodes in HMS. This module does not
fabricate them. It returns only the 20 available channels, in their
original relative EEGDM order, plus the indices that were dropped so
callers can correctly subset any pretrained per-channel embedding
table (e.g. EEGDM's label_embed) rather than discarding it outright.

This module performs ONLY the bipolar subtraction. It does not filter,
notch, resample, or scale -- those remain the caller's responsibility,
applied to the monopolar signal BEFORE calling build_bipolar(), matching
original EEGDM's own order (raw.filter() / raw.notch_filter() happen
before mne.set_bipolar_reference() in EEGDM_original/src/preprocessing.py).
Bipolar subtraction is linear, so filter-then-subtract is exactly
equivalent to subtract-then-filter; this module preserves the original
order for behavioral fidelity, not because the order changes the math.
"""

EEGDM_BIPOLAR_ORDER_22 = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "A1-T3", "T3-C3", "C3-CZ", "C4-CZ",
    "T4-C4", "A2-T4",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

HMS_ELECTRODES = {
    "FP1": "Fp1", "F3": "F3", "C3": "C3", "P3": "P3",
    "F7": "F7", "T3": "T3", "T5": "T5", "O1": "O1",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
    "FP2": "Fp2", "F4": "F4", "C4": "C4", "P4": "P4",
    "F8": "F8", "T4": "T4", "T6": "T6", "O2": "O2",
}


def check_availability(bipolar_order=EEGDM_BIPOLAR_ORDER_22, electrodes=HMS_ELECTRODES):
    report = []
    for i, name in enumerate(bipolar_order):
        anode_name, cathode_name = name.split("-")
        anode_available = anode_name in electrodes
        cathode_available = cathode_name in electrodes
        report.append({
            "index": i,
            "name": name,
            "anode": anode_name,
            "cathode": cathode_name,
            "anode_available": anode_available,
            "cathode_available": cathode_available,
            "available": anode_available and cathode_available,
        })
    return report


def build_bipolar(sig, channel_names, bipolar_order=EEGDM_BIPOLAR_ORDER_22,
                   electrodes=HMS_ELECTRODES):
    name_to_row = {n: i for i, n in enumerate(channel_names)}
    availability = check_availability(bipolar_order, electrodes)

    bipolar_rows = []
    used_names = []
    used_indices = []
    skipped = []

    for entry in availability:
        if not entry["available"]:
            skipped.append(entry)
            continue

        anode_hms_name = electrodes[entry["anode"]]
        cathode_hms_name = electrodes[entry["cathode"]]

        if anode_hms_name not in name_to_row or cathode_hms_name not in name_to_row:
            entry = dict(entry)
            entry["available"] = False
            entry["reason"] = (
                f"'{anode_hms_name}' or '{cathode_hms_name}' not found in "
                f"this file's actual channel_names"
            )
            skipped.append(entry)
            continue

        anode_row = sig[name_to_row[anode_hms_name]]
        cathode_row = sig[name_to_row[cathode_hms_name]]
        bipolar_rows.append(anode_row - cathode_row)
        used_names.append(entry["name"])
        used_indices.append(entry["index"])

    import numpy as np
    bipolar_sig = np.stack(bipolar_rows, axis=0) if bipolar_rows else np.zeros((0, sig.shape[1]), dtype=sig.dtype)

    return bipolar_sig, used_names, used_indices, skipped


if __name__ == "__main__":
    report = check_availability()
    print(f"{'idx':<4} {'channel':<10} {'anode':<6} {'cathode':<8} {'available'}")
    for e in report:
        print(f"{e['index']:<4} {e['name']:<10} {e['anode']:<6} {e['cathode']:<8} {e['available']}")

    available = [e for e in report if e["available"]]
    missing = [e for e in report if not e["available"]]
    print(f"\n{len(available)} / {len(report)} EEGDM bipolar channels constructible from HMS.")
    if missing:
        print("Missing (cannot construct, not fabricated):")
        for e in missing:
            missing_electrode = e["anode"] if not e["anode_available"] else e["cathode"]
            print(f"  {e['name']}: requires anode={e['anode']}, cathode={e['cathode']} "
                  f"-- missing electrode: {missing_electrode}")
