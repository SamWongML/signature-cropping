"""End-to-end pipeline runner shared by REST and MCP.

One implementation, two callers — matches CLAUDE.md's "one way to do each
thing" rule.
"""

from __future__ import annotations

import time
import uuid

from sigcrop.api.schemas import (
    BBox,
    BBoxNorm,
    CropOptions,
    CropResponse,
    ReturnFormat,
    SignatureResult,
    TimingMs,
)
from sigcrop.pipeline.detector import get_detector
from sigcrop.pipeline.encode import EncodedSignature, encode_signatures
from sigcrop.pipeline.heuristics import find_candidate_regions
from sigcrop.pipeline.ingest import ingest
from sigcrop.pipeline.postprocess import postprocess_detections
from sigcrop.pipeline.preprocess import preprocess_page


def run_pipeline(
    data: bytes,
    mime_hint: str | None,
    options: CropOptions,
    request_id: str,
) -> CropResponse:
    detector = get_detector()
    detector.warm_up()

    pre_ms = 0.0
    inf_ms = 0.0
    post_ms = 0.0
    encoded: list[EncodedSignature] = []

    t0 = time.perf_counter()
    doc = ingest(data, mime_hint=mime_hint)
    pre_ms += (time.perf_counter() - t0) * 1000.0

    for page_index, page_bgr in enumerate(doc.pages):
        t_pre = time.perf_counter()
        pp = preprocess_page(page_bgr, page_index=page_index)
        candidates = find_candidate_regions(pp.src_bgr)
        pre_ms += (time.perf_counter() - t_pre) * 1000.0

        if not candidates:
            continue

        t_inf = time.perf_counter()
        detections = detector.infer(pp.model_input)
        inf_ms += (time.perf_counter() - t_inf) * 1000.0

        t_post = time.perf_counter()
        sigs = postprocess_detections(
            detections,
            page_bgr=pp.src_bgr,
            letterbox=pp.letterbox,
            page_index=page_index + 1,
            confidence_threshold=options.confidence_threshold,
            nms_iou=0.5,
            padding_pct=options.padding_pct,
            apply_mask=options.apply_mask,
        )
        encoded_page = encode_signatures(
            sigs,
            return_format=options.return_format.value,
            s3_prefix=options.s3_prefix,
        )
        encoded.extend(encoded_page)
        post_ms += (time.perf_counter() - t_post) * 1000.0

    results: list[SignatureResult] = []
    for enc in encoded:
        page_idx = enc.page - 1
        if 0 <= page_idx < len(doc.pages):
            ph, pw = doc.pages[page_idx].shape[:2]
        else:
            pw = ph = 1
        results.append(
            SignatureResult(
                id=f"sig_{uuid.uuid4().hex}",
                page=enc.page,
                bbox=BBox(x=enc.x, y=enc.y, w=enc.w, h=enc.h),
                bbox_normalized=BBoxNorm(
                    x=enc.x / max(1, pw),
                    y=enc.y / max(1, ph),
                    w=enc.w / max(1, pw),
                    h=enc.h / max(1, ph),
                ),
                confidence=enc.confidence,
                crop_b64=enc.crop_b64,
                crop_s3_uri=enc.crop_s3_uri,
                mask_applied=enc.mask_applied,
            )
        )

    return CropResponse(
        request_id=request_id,
        model_version=detector.model_version,
        page_count=len(doc.pages),
        signatures=results,
        timing_ms=TimingMs(
            preprocess=int(pre_ms),
            inference=int(inf_ms),
            postprocess=int(post_ms),
        ),
    )


def run_pipeline_regions_only(
    data: bytes,
    mime_hint: str | None,
    options: CropOptions,
    request_id: str,
) -> CropResponse:
    """Variant that skips PNG encoding — coordinates only."""
    skinny = options.model_copy(update={"return_format": ReturnFormat.INLINE_B64})
    full = run_pipeline(data, mime_hint, skinny, request_id)
    stripped = [s.model_copy(update={"crop_b64": None}) for s in full.signatures]
    return full.model_copy(update={"signatures": stripped})
