"""
gst_qdrant_delete.py
Remove all chunks that came from one PDF out of Qdrant (no re-embedding needed).

Usage:
    python gst_qdrant_delete.py cgst_rules_partB_2017.pdf --dry-run     only count them
    python gst_qdrant_delete.py cgst_rules_partB_2017.pdf               delete them
"""

import argparse
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

COLLECTION = "gst_law"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    client = QdrantClient(url=os.getenv("QDRANT_URL", "").strip(),
                          api_key=os.getenv("QDRANT_API_KEY", "").strip(), timeout=120)

    # a filter on file_name needs a keyword index; creating it twice is harmless
    client.create_payload_index(COLLECTION, field_name="file_name",
                                field_schema=models.PayloadSchemaType.KEYWORD)
    flt = models.Filter(must=[models.FieldCondition(key="file_name",
                                                    match=models.MatchValue(value=args.file_name))])
    n = client.count(COLLECTION, count_filter=flt, exact=True).count
    print(f"{n} chunks in Qdrant come from {args.file_name}")
    if args.dry_run or n == 0:
        return
    client.delete(COLLECTION, points_selector=models.FilterSelector(filter=flt), wait=True)
    total = client.count(COLLECTION, exact=True).count
    print(f"Deleted. The collection now holds {total} chunks.")


if __name__ == "__main__":
    main()
