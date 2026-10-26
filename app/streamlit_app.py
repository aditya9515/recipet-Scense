"""ReceiptSense local review desk. Run with: streamlit run app/streamlit_app.py"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from receiptsense.common import FIELDS, ROOT, digest, read_json
from receiptsense.contracts import ImageRecord, OCRRecord, PredictionRecord
from receiptsense.dataset import dataset_status
from receiptsense.export import csv_export, json_export
from receiptsense.imaging import COLORS, evidence_image, retry_region_image
from receiptsense.model import trained_bundle_available
from receiptsense.pipeline import PipelineResult, process_receipt
from receiptsense.review_store import ReviewStore
from receiptsense.samples import create_samples

st.set_page_config(page_title="ReceiptSense · Review desk", page_icon="🧾", layout="wide")
st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Libre+Caslon+Display&display=swap');
:root {--ink:#183a34;--muted:#6c7e79;--paper:#f7f8f4;--teal:#237d68;}
.stApp {background:var(--paper);font-family:'DM Sans',sans-serif;color:var(--ink);}
.block-container {max-width:1360px;padding-top:2.7rem;padding-bottom:3rem;}
h1,h2,h3 {color:var(--ink);letter-spacing:-.025em;}
h1 {font-family:'Libre Caslon Display',Georgia,serif;font-size:3.5rem!important;font-weight:400!important;}
[data-testid="stSidebar"] {background:#edf1eb;border-right:1px solid #dde4db;}
[data-testid="stSidebar"] .block-container {padding-top:2rem;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#dce5dc!important;border-radius:14px!important;}
.eyebrow {text-transform:uppercase;letter-spacing:.2em;font-size:.7rem;color:#6f847b;font-weight:700;}
.lead {font-size:1rem;color:#6c7e79;line-height:1.7;margin-top:-.6rem;margin-bottom:1.8rem;}
.brand {font-family:Georgia,serif;color:#214f42;font-size:1.65rem;margin-bottom:.15rem;}
.brand-note {color:#7c8e83;font-size:.75rem;letter-spacing:.08em;margin-bottom:2rem;}
.field-tag {font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;font-weight:700;color:#72877d;margin-top:.6rem;}
.quiet-note {color:#7c8b83;font-size:.78rem;line-height:1.65;}
[data-testid="stMetricValue"] {font-family:Georgia,serif;color:#236c58;}
button[kind="primary"] {background:#267f69;border-color:#267f69;}
.stTextInput input,.stTextArea textarea {background:#fffefa!important;}
header[data-testid="stHeader"] {background:transparent;}
@media(max-width:760px) {
  [data-testid="stHorizontalBlock"] {flex-direction:column!important;}
  [data-testid="stColumn"] {width:100%!important;flex:1 1 auto!important;}
  h1 {font-size:2.7rem!important;}
}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def store():
    return ReviewStore()


def restore_prediction(prediction_id: str):
    folder = ROOT / "artifacts/predictions" / prediction_id
    st.session_state["result"] = PipelineResult(
        ImageRecord.model_validate(read_json(folder / "image.json")),
        OCRRecord.model_validate(read_json(folder / "ocr.json")),
        PredictionRecord.model_validate(read_json(folder / "prediction.json")),
        True,
    )
    st.session_state["page"] = "Scan receipt"
    st.session_state["restored"] = True
    latest = store().latest_review(prediction_id)
    st.session_state["review"] = latest
    prediction = st.session_state["result"].prediction
    for name in FIELDS:
        value = latest.values[name] if latest else prediction.fields[name].raw_value
        st.session_state[f"edit_{prediction_id}_{name}"] = value or ""


with st.sidebar:
    st.markdown(
        '<div class="brand">▤ ReceiptSense</div><div class="brand-note">THE RECEIPT REVIEW DESK</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Workspace", ["Scan receipt", "Review history", "Experiments", "Model & data"], key="page"
    )
    st.divider()
    st.markdown("**A clearer record.**")
    st.caption("Extract four receipt fields, inspect their source, and keep your corrections together.")
    st.markdown(
        '<div class="quiet-note">LOCAL WORKSPACE<br>Your receipt processing and review records stay on this computer.</div>',
        unsafe_allow_html=True,
    )


def scan_page():
    st.markdown('<div class="eyebrow">Capture · Understand · Review</div>', unsafe_allow_html=True)
    st.title("From receipt to record.")
    st.markdown(
        '<div class="lead">Bring a receipt. Check the details. Leave with a useful record.</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        upload_col, options_col = st.columns([1.8, 1], gap="large")
        with upload_col:
            source = st.radio("Receipt source", ["Upload a receipt", "Try an example"], horizontal=True)
            data, filename = None, None
            if source == "Upload a receipt":
                upload = st.file_uploader(
                    "Drop a receipt here",
                    type=["png", "jpg", "jpeg"],
                    help="One printed receipt. Up to 10 MB and 40 megapixels.",
                )
                if upload:
                    data, filename = upload.getvalue(), upload.name
            else:
                paths = (
                    create_samples()
                    if not (ROOT / "examples/receipt_clear.png").exists()
                    else sorted((ROOT / "examples").glob("receipt_*.png"))
                )
                selected = st.selectbox(
                    "Example receipt",
                    paths,
                    format_func=lambda p: "Clear receipt" if "clear" in p.name else "Ambiguous date",
                )
                data, filename = selected.read_bytes(), selected.name
                st.caption("Constructed examples for trying the workflow. They are not benchmark results.")
        with options_col:
            trained = trained_bundle_available(ROOT / "artifacts/models/receiptsense-layoutlmv3")
            options = ["Rules baseline", "LayoutLMv3"] if trained else ["Rules baseline"]
            method = st.selectbox("Extraction method", options)
            variant = st.selectbox("Image preparation", ["Original", "Deskew", "Mild contrast"])
            enhancements = {
                "Baseline": "none",
                "Address reconstruction": "address",
                "Selective recovery": "retry",
                "Combined": "combined",
            }
            enhancement = st.selectbox("Processing mode", list(enhancements))
            st.caption("Enhanced modes allow at most two extra cropped OCR calls. Scores are uncalibrated.")
            st.caption("The original is retained for evidence inspection.")
            if st.button("Read receipt", type="primary", width="stretch", disabled=data is None):
                try:
                    with st.spinner("Reading text and preparing the four fields…"):
                        result = process_receipt(
                            data,
                            filename,
                            method="rules" if method == "Rules baseline" else "layoutlmv3",
                            variant={"Original": "identity", "Deskew": "deskew", "Mild contrast": "contrast"}[
                                variant
                            ],
                            enhancement=enhancements[enhancement],
                        )
                        store().save_prediction(result.prediction)
                        st.session_state["result"] = result
                        st.session_state["input_hash"] = digest(data)
                        st.session_state.pop("review", None)
                        st.session_state["restored"] = False
                except Exception as exc:
                    st.error(str(exc))
    result = st.session_state.get("result")
    if result is None:
        a, b, c = st.columns(3)
        for col, number, heading, description in [
            (a, "01", "Read the receipt", "Printed text becomes an editable record."),
            (b, "02", "Follow the evidence", "Select a field to see its source region."),
            (c, "03", "Make it yours", "Correct, confirm, and export JSON or CSV."),
        ]:
            with col:
                st.markdown(f"### {number} · {heading}")
                st.caption(description)
        return
    if st.session_state.get("restored"):
        st.info(
            f"Viewing saved record: {result.image.filename}. Read receipt creates a new record from the selected input."
        )
    if (
        data is not None
        and digest(data) != st.session_state.get("input_hash")
        and not st.session_state.get("restored")
    ):
        st.info("A different receipt is selected. Click Read receipt to prepare its record.")
        return
    prediction = result.prediction
    review = st.session_state.get("review")
    if review and review.prediction_id != prediction.prediction_id:
        review = None
    st.divider()
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        title, view = st.columns([1, 1])
        title.subheader("Source receipt")
        frame = view.selectbox("View", ["Original", "Processed"], label_visibility="collapsed")
        field = st.radio(
            "Highlight evidence for",
            list(FIELDS),
            horizontal=True,
            format_func=str.title,
            key=f"evidence_{prediction.prediction_id}",
        )
        overlay = evidence_image(
            result.image,
            result.ocr,
            prediction.fields[field].evidence_unit_ids,
            frame="display" if frame == "Original" else "processed",
            color=COLORS[field],
        )
        st.image(overlay, width="stretch")
        st.caption(
            f"{result.image.filename} · {result.image.display_size[0]} × {result.image.display_size[1]} px · {len(result.ocr.units)} OCR units"
        )
        if any(u.box_source == "replicated_line" for u in result.ocr.units):
            st.caption("Evidence uses coarse line regions where the OCR engine supplied line geometry.")
        if prediction.recovery and prediction.recovery.attempts:
            with st.expander("OCR retry regions"):
                st.image(
                    retry_region_image(result.image, [a.crop_bbox for a in prediction.recovery.attempts]),
                    width="stretch",
                )
                st.caption(
                    "Regions are mapped back to the original receipt. Rejected attempts remain auditable."
                )
    with right:
        st.subheader("Receipt details")
        flagged = sum(bool(f.review_reasons) or f.status != "predicted" for f in prediction.fields.values())
        st.caption(
            f"{'LayoutLMv3' if prediction.method == 'layoutlmv3' else 'Rules baseline'} · {flagged} fields to check"
        )
        if prediction.recovery:
            recovery = prediction.recovery
            st.caption(
                f"{recovery.mode.title()} · {recovery.logical_attempts} retry attempts · {recovery.uncached_ocr_calls} uncached retry OCR calls · {recovery.elapsed_seconds:.2f}s extra"
            )
            with st.expander("Before / after and recovery decisions"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Field": f,
                                "Before": recovery.baseline_fields[f].raw_value,
                                "After": prediction.fields[f].raw_value,
                            }
                            for f in FIELDS
                        ]
                    ),
                    hide_index=True,
                )
                if recovery.address_decision:
                    decision = recovery.address_decision
                    st.caption(
                        f"Address reconstruction: {decision['outcome']} · {decision['reason'].replace('_', ' ')}"
                    )
                for attempt in recovery.attempts:
                    st.markdown(f"**{attempt.attempt_id} · {attempt.field.title()} · {attempt.outcome}**")
                    st.caption(attempt.reason.replace("_", " "))
                    st.write(f"Candidate: {attempt.candidate or 'No readable candidate'}")
                    st.caption(
                        f"{attempt.preparation.title()} crop · {'Cached OCR' if attempt.cache_hit else 'OCR call' if attempt.ocr_called else 'No OCR call'} · {attempt.elapsed_seconds:.2f}s"
                    )
        values = {}
        labels = {"company": "Company", "address": "Address", "date": "Date", "total": "Total amount"}
        for name in FIELDS:
            item = prediction.fields[name]
            initial = review.values[name] if review else item.raw_value
            key = f"edit_{prediction.prediction_id}_{name}"
            if key not in st.session_state:
                st.session_state[key] = initial or ""
            if name == "address":
                value = st.text_area(labels[name], key=key, height=110)
            else:
                value = st.text_input(labels[name], key=key)
            values[name] = value.strip() or None
            if item.review_reasons:
                human_reasons = ", ".join(reason.replace("_", " ") for reason in item.review_reasons)
                st.caption(f"Check: {human_reasons}.")
            if name in {"date", "total"} and item.normalized_value and not review:
                st.caption(f"Normalized: {item.normalized_value}")
        save, confirm = st.columns(2)
        if save.button("Save changes", width="stretch"):
            review = store().save_review(prediction.prediction_id, values)
            st.session_state["review"] = review
        if confirm.button("Confirm record", type="primary", width="stretch"):
            review = store().save_review(prediction.prediction_id, values, confirm=True)
            st.session_state["review"] = review
        persisted_values = (
            review.values if review else {name: prediction.fields[name].raw_value for name in FIELDS}
        )
        unsaved = values != persisted_values
        if unsaved:
            st.warning("You have unsaved changes. Save or confirm them before exporting.")
        elif review:
            st.success(
                "Record confirmed."
                if review.state == "confirmed"
                else "Changes saved. You can confirm when ready."
            )
            unresolved = [name for name, reasons in review.review_reasons.items() if reasons]
            if unresolved:
                st.caption(
                    "Raw text retained; normalized values remain unresolved for: " + ", ".join(unresolved)
                )
        st.markdown("#### Export record")
        st.caption(
            f"Version: {review.state if review and not unsaved else 'unreviewed'} · Original machine prediction is preserved."
        )
        json_col, csv_col = st.columns(2)
        json_col.download_button(
            "Download JSON",
            json_export(prediction, review),
            file_name=f"receipt_{prediction.receipt_id}.json",
            mime="application/json",
            disabled=unsaved,
            width="stretch",
        )
        csv_col.download_button(
            "Download CSV",
            csv_export(prediction, review),
            file_name=f"receipt_{prediction.receipt_id}.csv",
            mime="text/csv",
            disabled=unsaved,
            width="stretch",
        )
        with st.expander("Processing details & review history"):
            st.json(
                {
                    "method": prediction.method,
                    "timings_seconds": prediction.timings,
                    "cache_hit": result.cache_hit,
                    "OCR": result.ocr.engine,
                    "review": review.model_dump(mode="json") if review else None,
                    "recovery": prediction.recovery.model_dump(mode="json") if prediction.recovery else None,
                }
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "text": u.text,
                            "line": u.line_id,
                            "region": u.unit_id,
                            "recognition_score": u.recognition_score,
                            "source": u.attempt_id,
                        }
                        for u in result.ocr.units
                    ]
                ),
                hide_index=True,
            )


def history_page():
    st.markdown('<div class="eyebrow">Your local records</div>', unsafe_allow_html=True)
    st.title("Review history.")
    rows = store().recent()
    if not rows:
        st.info("Read a receipt to create your first record.")
    for row in rows:
        with st.container(border=True):
            a, b = st.columns([4, 1])
            latest = row.get("latest_review")
            values = latest["values"] if latest else {n: f["raw_value"] for n, f in row["fields"].items()}
            a.markdown(f"**{values['company'] or 'Unidentified company'}**")
            a.caption(
                f"{values['date'] or 'No date'} · {values['total'] or 'No total'} · {row['method']} · {latest['state'] if latest else 'unreviewed'}"
            )
            b.button(
                "Open record",
                key=row["prediction_id"],
                on_click=restore_prediction,
                args=(row["prediction_id"],),
            )


def experiments_page():
    st.markdown('<div class="eyebrow">Measured evidence</div>', unsafe_allow_html=True)
    st.title("Experiment notebook.")
    summaries = sorted((ROOT / "artifacts/runs").glob("*/summary.json"))
    summaries += sorted((ROOT / "artifacts/novelty").glob("*/summary.json"))
    local_names = {p.parent.name for p in summaries}
    summaries += [
        p
        for p in sorted((ROOT / "results/novelty").glob("*/summary.json"))
        if p.parent.name not in local_names
    ]
    if not summaries:
        st.info(
            "No evaluation runs yet. Prepared results will appear here after the evaluation command runs."
        )
        return
    selected = st.selectbox("Saved run", summaries, format_func=lambda p: p.parent.name)
    summary = read_json(selected)
    st.caption(
        f"{summary['n_receipts']} receipts · {summary['created_at']} · {summary.get('latency_scope', '')}"
    )
    if summary.get("purpose"):
        st.info(summary["purpose"])
    policy = st.radio("Metric policy", ["strict-v1", "normalized-v1"], horizontal=True)
    chart_rows = []
    for method, results in summary["results"].items():
        metrics = results[policy]
        st.subheader(method.title())
        a, b, c = st.columns(3)
        a.metric("Macro field exact match", f"{metrics['macro_field_em']:.1%}")
        b.metric("All four fields correct", f"{metrics['all_four_em']:.1%}")
        c.metric("Missing predictions", metrics["missing_predictions"])
        for name, metric in metrics["per_field"].items():
            chart_rows.append({"Field": name, "Method": method, "Exact match": metric["exact_match"]})
    table = pd.DataFrame(chart_rows)
    st.bar_chart(table.pivot(index="Field", columns="Method", values="Exact match"), stack=False)
    if summary.get("configuration_hash"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Condition": condition,
                        "Address token F1": values["address_token_f1"],
                        "Retry rate": values["retry_rate"],
                        "Logical attempts": values["logical_retry_attempts"],
                        "Uncached retry calls": values["uncached_retry_ocr_calls"],
                        "Errors corrected": sum(values["corrected_fields"].values()),
                        "Errors introduced": sum(values["introduced_errors"].values()),
                    }
                    for condition, values in summary["results"].items()
                ]
            ),
            hide_index=True,
        )
    with st.expander("Complete saved result and provenance"):
        st.json(summary)


def model_page():
    st.markdown('<div class="eyebrow">Reproducibility workspace</div>', unsafe_allow_html=True)
    st.title("Model & data.")
    st.subheader("Dataset audit")
    st.json(dataset_status())
    path = ROOT / "artifacts/models/receiptsense-layoutlmv3/receiptsense_model.json"
    if path.exists():
        metadata = read_json(path)
        st.subheader("Receipt extractor")
        st.caption(metadata.get("adaptation", "LayoutLMv3 adaptation"))
        if metadata.get("history"):
            st.line_chart(pd.DataFrame(metadata["history"]).set_index("epoch")[["validation_macro_field_em"]])
        with st.expander("Training manifest"):
            st.json(metadata)
    else:
        st.info("Rules extraction is available. A trained LayoutLMv3 bundle has not been installed yet.")
    st.markdown(
        "The implementation and experiment commands are documented in **RUNNING.md** in the project folder."
    )


{
    "Scan receipt": scan_page,
    "Review history": history_page,
    "Experiments": experiments_page,
    "Model & data": model_page,
}[page]()
