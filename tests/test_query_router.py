"""Tests for query router module."""

import pytest
from src.routers.query_router import (
    RouteType,
    classify_query,
    has_direct_calculation_intent,
    has_legal_intent,
    has_rate_intent,
)


@pytest.mark.parametrize(
    "query",
    [
        "What is 5% of 100000?",
        "If GST amount is 5000 at 5%, what is taxable value?",
        "If I have 5000 GST credits and I charge 5%, then how much taxable amount?",
        (
            "કોઈ વસ્તુની કિંમત ₹20,000 છે. પહેલા કિંમત પર 10% ડિસ્કાઉન્ટ આપવામાં "
            "આવે છે અને પછી બાકી રકમ પર 5% GST લગાવવામાં આવે છે. અંતિમ રકમ કેટલી થશે?"
        ),
    ],
)
def test_direct_calculation_queries_classified_correctly(query: str):
    assert classify_query(query) == RouteType.DIRECT
    assert has_direct_calculation_intent(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What is the GST rate on chocolate?",
        "HSN 8471",
        "SAC 9963 rate",
        "tax on milk",
        "cess on motor vehicles",
        "what is the rate of footwear?",
        "8471",
        "Heading 9954",
        "What is the tax slab for restaurant services?",
        "applicable rate for hotel accommodation",
        "18% GST items",
        "Give me full GST details for fresh milk.",
    ],
)
def test_rate_queries_classified_correctly(query: str):
    assert classify_query(query) == RouteType.RATE
    assert has_rate_intent(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "How can GST registration be cancelled?",
        "Grounds for cancellation under Rule 21",
        "Which form is used to apply for cancellation of registration?",
        "What is the procedure for revocation of cancellation?",
        "Time limit to reply to show cause notice under Section 73",
        "Who is the proper officer under CGST Act 2017?",
        "What are the penalties for late filing of returns?",
        "Can registration be suspended before cancellation?",
        "Appeals process before the Appellate Tribunal",
        "CGST Act 2017",
    ],
)
def test_legal_queries_classified_correctly(query: str):
    assert classify_query(query) == RouteType.LEGAL
    assert has_legal_intent(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What is the GST rate on computer parts under HSN 8471 and how do I claim ITC under Section 16?",
        "GST rate on footwear and procedure for cancellation of registration under Rule 22",
        "What is the rate of tax on ice cream and what are the conditions for composition levy under section 10?",
        "Rate for works contract and is registration mandatory under section 24?",
        "What is the cess on motor vehicles and what penalty applies under section 122?",
    ],
)
def test_mixed_queries_classified_correctly(query: str):
    assert classify_query(query) == RouteType.MIXED
    assert has_rate_intent(query) is True
    assert has_legal_intent(query) is True


def test_empty_and_fallback_queries():
    assert classify_query("") == RouteType.LEGAL
    assert classify_query("   ") == RouteType.LEGAL
    assert classify_query("Hello there") == RouteType.LEGAL


def test_document_dependent_calculation_uses_rate_route():
    query = "Using the applicable GST rate from the documents, calculate tax on 100000"
    assert classify_query(query) == RouteType.RATE
    assert has_direct_calculation_intent(query) is False


def test_gujarati_applicable_rate_query_does_not_use_direct_route():
    query = "દસ્તાવેજો મુજબ આ વસ્તુ પર લાગુ GST દર કેટલો છે?"
    assert classify_query(query) == RouteType.RATE
    assert has_direct_calculation_intent(query) is False
