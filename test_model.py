"""
Test the LLM receipt extractor on a single image.

Usage:
    python test_model_fixed.py path/to/receipt.jpg
    python test_model_fixed.py path/to/receipt.jpg --json
"""

import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test LLM receipt extraction")
    parser.add_argument("image", help="Path to receipt image")
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    print(f"[INFO] Loading LLM extractor...")
    from model_llm import LLMReceiptExtractor
    extractor = LLMReceiptExtractor()

    print(f"[INFO] Analysing {image_path.name}...")
    import time
    t0 = time.perf_counter()
    result = extractor.extract_from_path(str(image_path))
    elapsed = time.perf_counter() - t0

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print()
    print("=" * 60)
    print(f"  STORE   : {result['store_name'] or '(not detected)'}")
    print(f"  ADDRESS : {result['store_address'] or '(not detected)'}")
    print(f"  DATE    : {result['date'] or '—'}  {result['time'] or ''}")
    print(f"  PAYMENT : {result['payment_method']}")
    print(f"  CONF    : {result['confidence']:.1f}%")
    print("-" * 60)
    if result["items"]:
        for item in result["items"]:
            qty = f"x{item['quantity']}" if item['quantity'] > 1 else ""
            price = f"€{item['total_price']:.2f}" if item['total_price'] else "  —  "
            discount = f" (-€{item['discount']:.2f})" if item['discount'] > 0 else ""
            print(f"  {item['name']:<35} {qty:<4} {price:>8}{discount}")
    else:
        print("  (no line items detected)")
    print("-" * 60)
    if result['savings'] > 0:
        print(f"  SAVINGS : €{result['savings']:.2f}")
    print(f"  SUBTOTAL: €{result['subtotal']:.2f}")
    print(f"  TAX     : €{result['tax']:.2f}")
    print(f"  TOTAL   : €{result['total']:.2f}")
    print("=" * 60)
    print(f"  Done in {elapsed:.2f}s")
    print()


if __name__ == "__main__":
    main()