"""Deterministic Arithmetic-Only Calculator for GST Assistant.

Performs purely mathematical calculations on structured numeric inputs.
Does NOT decide legal rules, ledger eligibility, utilization order, or invent tax rates.
All statutory determinations must happen before calculation inputs are provided.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CalculationInputs(BaseModel):
    """Structured numeric calculation inputs provided to the calculator."""
    model_config = ConfigDict(extra="ignore")

    operation: str  # "max_taxable_value_from_credit", "tax_on_value", "discount_and_tax", "arithmetic"
    available_eligible_credit: Optional[float] = None
    tax_rate_pct: Optional[float] = None
    taxable_value: Optional[float] = None
    discount_pct: Optional[float] = None
    base_amount: Optional[float] = None
    credit_breakdown: dict[str, float] = Field(default_factory=dict)
    custom_values: dict[str, float] = Field(default_factory=dict)

    @field_validator("credit_breakdown", "custom_values", mode="before")
    @classmethod
    def _ensure_dict(cls, v: Any) -> dict[str, Any]:
        if v is None or not isinstance(v, dict):
            return {}
        return v


class CalculationResult(BaseModel):
    """Deterministic calculation output with transparent arithmetic steps."""
    model_config = ConfigDict(extra="ignore")

    operation: str
    status: str = "success"  # "success" | "error"
    result_value: Optional[float] = None
    formula: str = ""
    steps: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None

    @field_validator("steps", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list[Any]:
        if v is None or not isinstance(v, list):
            return []
        return v


def calculate_max_taxable_value_from_credit(
    eligible_credit: float,
    tax_rate_pct: float,
    credit_breakdown: dict[str, float] | None = None,
) -> CalculationResult:
    """Calculate maximum taxable value covered by available eligible credit: value = credit / rate."""
    if tax_rate_pct <= 0:
        return CalculationResult(
            operation="max_taxable_value_from_credit",
            status="error",
            formula="taxable_value = eligible_credit / (tax_rate_pct / 100)",
            error_message="Tax rate must be strictly greater than 0% to compute taxable value from credit.",
        )
    if eligible_credit < 0:
        return CalculationResult(
            operation="max_taxable_value_from_credit",
            status="error",
            formula="taxable_value = eligible_credit / (tax_rate_pct / 100)",
            error_message="Eligible credit cannot be negative.",
        )

    rate_decimal = tax_rate_pct / 100.0
    taxable_val = round(eligible_credit / rate_decimal, 2)

    steps = []
    if credit_breakdown:
        parts = [f"{k.upper()}: ₹{v:,.2f}" for k, v in credit_breakdown.items() if v > 0]
        steps.append(f"Eligible credit breakdown: {', '.join(parts)}")
    steps.append(f"Total eligible input tax credit: ₹{eligible_credit:,.2f}")
    steps.append(f"Applicable tax rate: {tax_rate_pct:g}% (rate multiplier: {rate_decimal})")
    steps.append(f"Arithmetic Formula: Taxable Value = Eligible Credit / Tax Rate = ₹{eligible_credit:,.2f} / {rate_decimal} = ₹{taxable_val:,.2f}")
    steps.append(f"Verification: ₹{taxable_val:,.2f} × {tax_rate_pct:g}% = ₹{round(taxable_val * rate_decimal, 2):,.2f} tax liability, discharging cash payable to ₹0.00.")

    return CalculationResult(
        operation="max_taxable_value_from_credit",
        status="success",
        result_value=taxable_val,
        formula=f"₹{eligible_credit:,.2f} / {rate_decimal} = ₹{taxable_val:,.2f}",
        steps=steps,
    )


def calculate_tax_on_value(
    taxable_value: float,
    tax_rate_pct: float,
) -> CalculationResult:
    """Calculate output tax on taxable value: tax = value * (rate / 100)."""
    rate_decimal = tax_rate_pct / 100.0
    tax_amount = round(taxable_value * rate_decimal, 2)
    total_amount = round(taxable_value + tax_amount, 2)

    steps = [
        f"Taxable value: ₹{taxable_value:,.2f}",
        f"Applicable GST rate: {tax_rate_pct:g}%",
        f"Arithmetic: Tax Amount = ₹{taxable_value:,.2f} × {rate_decimal} = ₹{tax_amount:,.2f}",
        f"Total Invoice Amount: ₹{taxable_value:,.2f} + ₹{tax_amount:,.2f} = ₹{total_amount:,.2f}",
    ]

    return CalculationResult(
        operation="tax_on_value",
        status="success",
        result_value=tax_amount,
        formula=f"₹{taxable_value:,.2f} × {tax_rate_pct:g}% = ₹{tax_amount:,.2f}",
        steps=steps,
    )


def calculate_discount_and_tax(
    base_amount: float,
    discount_pct: float,
    tax_rate_pct: float,
) -> CalculationResult:
    """Calculate discounted amount then add GST."""
    discount_decimal = discount_pct / 100.0
    discount_amt = round(base_amount * discount_decimal, 2)
    taxable_amt = round(base_amount - discount_amt, 2)
    tax_decimal = tax_rate_pct / 100.0
    tax_amt = round(taxable_amt * tax_decimal, 2)
    final_amt = round(taxable_amt + tax_amt, 2)

    steps = [
        f"Base amount: ₹{base_amount:,.2f}",
        f"Discount ({discount_pct:g}%): ₹{base_amount:,.2f} × {discount_decimal} = ₹{discount_amt:,.2f}",
        f"Taxable amount after discount: ₹{base_amount:,.2f} - ₹{discount_amt:,.2f} = ₹{taxable_amt:,.2f}",
        f"GST ({tax_rate_pct:g}%): ₹{taxable_amt:,.2f} × {tax_decimal} = ₹{tax_amt:,.2f}",
        f"Final amount payable: ₹{taxable_amt:,.2f} + ₹{tax_amt:,.2f} = ₹{final_amt:,.2f}",
    ]

    return CalculationResult(
        operation="discount_and_tax",
        status="success",
        result_value=final_amt,
        formula=f"(₹{base_amount:,.2f} - {discount_pct:g}%) + {tax_rate_pct:g}% GST = ₹{final_amt:,.2f}",
        steps=steps,
    )


def execute_calculator(inputs: CalculationInputs) -> CalculationResult:
    """Dispatch structured calculation input to the appropriate deterministic formula."""
    op = inputs.operation
    if op == "max_taxable_value_from_credit":
        if inputs.available_eligible_credit is None or inputs.tax_rate_pct is None:
            return CalculationResult(
                operation=op,
                status="error",
                error_message="Both available_eligible_credit and tax_rate_pct are required for max_taxable_value_from_credit.",
            )
        return calculate_max_taxable_value_from_credit(
            eligible_credit=inputs.available_eligible_credit,
            tax_rate_pct=inputs.tax_rate_pct,
            credit_breakdown=inputs.credit_breakdown,
        )

    if op == "tax_on_value":
        if inputs.taxable_value is None or inputs.tax_rate_pct is None:
            return CalculationResult(
                operation=op,
                status="error",
                error_message="Both taxable_value and tax_rate_pct are required for tax_on_value.",
            )
        return calculate_tax_on_value(
            taxable_value=inputs.taxable_value,
            tax_rate_pct=inputs.tax_rate_pct,
        )

    if op == "discount_and_tax":
        if inputs.base_amount is None or inputs.discount_pct is None or inputs.tax_rate_pct is None:
            return CalculationResult(
                operation=op,
                status="error",
                error_message="base_amount, discount_pct, and tax_rate_pct are required for discount_and_tax.",
            )
        return calculate_discount_and_tax(
            base_amount=inputs.base_amount,
            discount_pct=inputs.discount_pct,
            tax_rate_pct=inputs.tax_rate_pct,
        )

    if op == "discount_only":
        if inputs.base_amount is None or inputs.discount_pct is None:
            return CalculationResult(
                operation=op,
                status="error",
                error_message="base_amount and discount_pct are required for discount_only.",
            )
        disc_dec = inputs.discount_pct / 100.0
        disc_val = round(inputs.base_amount * disc_dec, 2)
        res_val = round(inputs.base_amount - disc_val, 2)
        return CalculationResult(
            operation="discount_only",
            status="success",
            result_value=res_val,
            formula=f"₹{inputs.base_amount:,.2f} - {inputs.discount_pct:g}% = ₹{res_val:,.2f}",
            steps=[
                f"Base amount: ₹{inputs.base_amount:,.2f}",
                f"Discount ({inputs.discount_pct:g}%): ₹{inputs.base_amount:,.2f} × {disc_dec} = ₹{disc_val:,.2f}",
                f"Discounted amount: ₹{inputs.base_amount:,.2f} - ₹{disc_val:,.2f} = ₹{res_val:,.2f}",
            ],
        )

    return CalculationResult(
        operation=op,
        status="error",
        error_message=f"Unsupported calculation operation: '{op}'",
    )
