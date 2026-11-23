"""
gst_setup.py
Creates the GST knowledge base folder structure and an empty manifest.csv.

How to run (on your computer):
    python gst_setup.py
It is safe to run again: existing folders and an existing manifest.csv are left untouched.
"""

import csv
from pathlib import Path

ROOT = Path("gst_knowledge_base")

# The four pillars, in your numbering
PILLARS = ["1_acts", "2_rules", "3_notifications", "4_circulars"]

# Central branches that exist under every pillar
CENTRAL_BRANCHES = ["cgst", "igst", "utgst"]

# States and UTs that have their own SGST law (they have a legislature)
SGST_STATES = [
    "andhra_pradesh", "arunachal_pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal_pradesh", "jharkhand", "karnataka",
    "kerala", "madhya_pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "punjab", "rajasthan", "sikkim",
    "tamil_nadu", "telangana", "tripura", "uttar_pradesh", "uttarakhand",
    "west_bengal",
    # Union Territories with a legislature (they also use SGST)
    "delhi", "puducherry", "jammu_and_kashmir",
]

# Columns of the manifest (one row per downloaded file)
MANIFEST_COLUMNS = [
    "file_name",      # name of the PDF as saved in your folder
    "gst_type",       # CGST / IGST / SGST / UTGST
    "doc_type",       # act / rule / notification / circular
    "state",          # "central" or the state name
    "number",         # official number (notifications and circulars only)
    "date",           # official date on the document, YYYY-MM-DD
    "amended_up_to",  # for Acts and Rules: "updated up to" date, YYYY-MM-DD
    "source_url",     # official website link you downloaded it from
]


def main():
    ROOT.mkdir(exist_ok=True)

    folder_count = 0
    for pillar in PILLARS:
        for branch in CENTRAL_BRANCHES:
            (ROOT / pillar / branch).mkdir(parents=True, exist_ok=True)
            folder_count += 1
        for state in SGST_STATES:
            (ROOT / pillar / "sgst" / state).mkdir(parents=True, exist_ok=True)
            folder_count += 1

    manifest = ROOT / "manifest.csv"
    if manifest.exists():
        print("manifest.csv already exists - left unchanged.")
    else:
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(MANIFEST_COLUMNS)
        print("Created manifest.csv with header row.")

    print(f"Done. {folder_count} folders ready under '{ROOT}/'.")


if __name__ == "__main__":
    main()
