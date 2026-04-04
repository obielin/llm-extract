"""
Example: Extract invoice data from a text document.

Run:
    python examples/invoice_extraction.py
"""

from pydantic import BaseModel, Field
from typing import Optional

from llm_extract import extract_text


class Invoice(BaseModel):
    vendor: str = Field(description="Name of the supplier or vendor")
    invoice_number: Optional[str] = Field(None, description="Invoice reference number")
    date: str = Field(description="Invoice date (ISO format preferred)")
    due_date: Optional[str] = Field(None, description="Payment due date")
    subtotal: Optional[float] = Field(None, description="Subtotal before tax in GBP")
    vat: Optional[float] = Field(None, description="VAT amount in GBP")
    total: float = Field(description="Total amount due in GBP")
    payment_terms: Optional[str] = Field(None, description="Payment terms e.g. 30 days net")


SAMPLE_INVOICE = """
INVOICE

From: Acme Software Solutions Ltd
      12 Tech Park, Leeds, LS1 1AA
      VAT Reg: GB123456789

To:   HMRC Digital Division
      100 Parliament Street, London

Invoice Number: INV-2026-0142
Invoice Date:   15 January 2026
Due Date:       14 February 2026

Description                          Qty    Unit Price    Total
─────────────────────────────────────────────────────────────
AI Development Consultancy (Jan)      10    £450.00      £4,500.00
Cloud Infrastructure Setup             1    £850.00        £850.00
─────────────────────────────────────────────────────────────
                                    Subtotal:            £5,350.00
                                    VAT (20%):           £1,070.00
                                    TOTAL DUE:           £6,420.00

Payment Terms: 30 days net
Bank: HSBC | Sort: 40-12-34 | Account: 12345678
"""


def main():
    print("llm-extract — Invoice Extraction Example")
    print("=" * 50)

    result = extract_text(SAMPLE_INVOICE, schema=Invoice)

    if result.success:
        print(f"\n✓ Extracted {result.fields_found}/{result.fields_total} fields")
        print(f"  Coverage: {result.coverage:.0%}")
        print(f"  Mean confidence: {result.mean_confidence:.2f}\n")

        print("Extracted Data:")
        for field, value in result.data.model_dump().items():
            if value is not None:
                conf = result.confidence.get(field, 0)
                source = result.sources.get(field, "")[:50]
                print(f"  {field:<20} {str(value):<25} conf={conf:.2f}  src='{source}'")

        low_conf = result.low_confidence_fields(threshold=0.8)
        if low_conf:
            print(f"\n⚠ Review these fields manually: {low_conf}")
    else:
        print("✗ Extraction failed")
        print(result.raw_response[:200])


if __name__ == "__main__":
    main()
