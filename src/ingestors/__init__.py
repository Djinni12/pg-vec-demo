"""Ingestors for loading GST data into PostgreSQL."""

from .ingest_gst import DATABASE_URL, MODEL_NAME, FIELDS, exact_codes, rate_percent, read_dataset, save_records
from .ingest_act_chunks import load_act_embeddings
from .ingest_rule_chunks import load_rule_embeddings  
from .ingest_form_chunks import load_form_embeddings

__all__ = [
    "DATABASE_URL",
    "MODEL_NAME",
    "FIELDS",
    "exact_codes",
    "rate_percent",
    "read_dataset",
    "save_records",
    "load_act_embeddings",
    "load_rule_embeddings",
    "load_form_embeddings",
]
