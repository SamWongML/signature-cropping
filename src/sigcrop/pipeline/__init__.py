"""Inference pipeline. Pure functions; no FastAPI/MCP imports."""

from sigcrop.pipeline.encode import encode_signatures
from sigcrop.pipeline.heuristics import find_candidate_regions
from sigcrop.pipeline.ingest import IngestedDocument, ingest
from sigcrop.pipeline.postprocess import postprocess_detections
from sigcrop.pipeline.preprocess import PreprocessedPage, preprocess_page
from sigcrop.pipeline.run import run_pipeline, run_pipeline_regions_only

__all__ = [
    "IngestedDocument",
    "PreprocessedPage",
    "encode_signatures",
    "find_candidate_regions",
    "ingest",
    "postprocess_detections",
    "preprocess_page",
    "run_pipeline",
    "run_pipeline_regions_only",
]
