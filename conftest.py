import sys
import os
import importlib

# Add the workspace root to Python path for imports
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

MODULE_ALIASES = {
    "act_chunker": "src.chunkers.act_chunker",
    "form_chunker": "src.chunkers.form_chunker",
    "rules_chunker": "src.chunkers.rules_chunker",
    "act_embedder": "src.embedders.act_embedder",
    "gst_act_parser": "src.parsers.gst_act_parser",
    "gst_forms_parser": "src.parsers.gst_forms_parser",
    "gst_rules_parser": "src.parsers.gst_rules_parser",
    "ingest_act_chunks": "src.ingestors.ingest_act_chunks",
    "ingest_form_chunks": "src.ingestors.ingest_form_chunks",
    "ingest_gst": "src.ingestors.ingest_gst",
    "ingest_rule_chunks": "src.ingestors.ingest_rule_chunks",
    "keyword_search": "src.retrievers.keyword_search",
    "legal_dense_retriever": "src.retrievers.legal_dense_retriever",
    "rrf": "src.retrievers.rrf",
    "vector_search": "src.retrievers.vector_search",
    "search_pgvec": "src.search_pgvec",
}

for alias, module_name in MODULE_ALIASES.items():
    sys.modules.setdefault(alias, importlib.import_module(module_name))
