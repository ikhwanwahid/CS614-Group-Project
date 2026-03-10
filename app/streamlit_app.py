"""Streamlit demo app for health claims fact-checker.

Run from project root:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work when launched
# via `streamlit run app/streamlit_app.py` from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Health Claims Fact-Checker", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VERDICT_COLORS = {
    "SUPPORTED": "green",
    "UNSUPPORTED": "red",
    "OVERSTATED": "orange",
    "INSUFFICIENT_EVIDENCE": "purple",
    "ERROR": "gray",
}

CHUNKING_STRATEGIES = ["fixed", "semantic", "section_aware", "recursive"]
RETRIEVAL_METHODS = ["naive", "hybrid", "hybrid_reranked"]
AGENT_ARCHITECTURES = ["single_pass", "langgraph_multi", "strands_rerouting"]
MODELS = ["claude-sonnet-4", "gpt-4o-mini", "claude-haiku", "llama-3.1-8b"]

DATA_DIR = _PROJECT_ROOT / "data"
RESULTS_DIR = _PROJECT_ROOT / "results" / "experiments"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_test_claims() -> list[dict]:
    """Load curated test claims from data/test_claims.json."""
    claims_path = DATA_DIR / "test_claims.json"
    if not claims_path.exists():
        return []
    with open(claims_path) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_experiment_configs() -> dict:
    """Import EXPERIMENT_CONFIGS from the experiment runner module."""
    try:
        from src.experiment_runner import EXPERIMENT_CONFIGS
        return EXPERIMENT_CONFIGS
    except ImportError:
        return {}


def load_experiment_results() -> dict[str, dict]:
    """Load all E*.json result files from results/experiments/."""
    results = {}
    if not RESULTS_DIR.exists():
        return results
    for path in sorted(RESULTS_DIR.glob("E*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            results[path.stem] = data
        except Exception:
            continue
    return results


def run_pipeline(claim: str, chunking: str, retrieval: str, agent: str, model: str) -> dict:
    """Run a single claim through the configurable pipeline."""
    from src.pipelines.configurable import run_experiment
    return run_experiment(
        claim=claim,
        chunking_strategy=chunking,
        retrieval_method=retrieval,
        agent_architecture=agent,
        model=model,
    )


def verdict_badge(verdict: str) -> str:
    """Return an HTML badge string for a verdict."""
    color = VERDICT_COLORS.get(verdict, "gray")
    return (
        f'<span style="background-color:{color};color:white;'
        f'padding:4px 12px;border-radius:4px;font-weight:bold;'
        f'font-size:0.95em;">{verdict}</span>'
    )


def display_result(result: dict, header: str | None = None):
    """Render a single fact-check result in the Streamlit UI."""
    if header:
        st.subheader(header)

    verdict = result.get("verdict", "N/A")
    st.markdown(verdict_badge(verdict), unsafe_allow_html=True)

    st.markdown("**Explanation**")
    st.write(result.get("explanation", "No explanation provided."))

    evidence = result.get("evidence", [])
    if evidence:
        st.markdown("**Evidence Passages**")
        for i, ev in enumerate(evidence, 1):
            source = ev.get("source", "Unknown")
            passage = ev.get("passage", ev.get("text", ""))
            relevance = ev.get("relevance_score", "")
            rel_str = f" (relevance: {relevance})" if relevance else ""
            st.markdown(f"> **[{i}]** *{source}*{rel_str}  \n> {passage}")
    else:
        st.info("No evidence passages returned.")

    metadata = result.get("metadata", {})
    if metadata:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Latency", f"{metadata.get('latency_seconds', 'N/A')} s")
        col_b.metric("Tokens", f"{metadata.get('total_tokens', 'N/A')}")
        col_c.metric("Est. Cost", f"${metadata.get('estimated_cost_usd', 0):.6f}")


# ---------------------------------------------------------------------------
# Header & disclaimer
# ---------------------------------------------------------------------------
st.title("Health Claims Fact-Checker")
st.caption("Not medical advice. Consult a healthcare professional.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_single, tab_compare, tab_dashboard = st.tabs(
    ["Single Check", "Compare", "Dashboard"]
)

# ============================= SINGLE CHECK ================================
with tab_single:
    st.header("Check a Health Claim")

    # --- Claim input ---
    test_claims = load_test_claims()
    example_options = ["(type your own)"] + [c["claim"][:100] for c in test_claims]
    selected_example = st.selectbox(
        "Select a curated example or type your own:",
        example_options,
        key="single_example",
    )

    if selected_example == "(type your own)":
        claim_text = st.text_area(
            "Enter a health claim:",
            height=80,
            placeholder="e.g., Vaccines cause autism",
            key="single_claim_input",
        )
    else:
        idx = example_options.index(selected_example) - 1
        claim_text = test_claims[idx]["claim"]
        expected = test_claims[idx].get("expected_verdict", "")
        st.text_area("Claim:", value=claim_text, height=80, disabled=True, key="single_claim_show")
        if expected:
            st.caption(f"Expected verdict: **{expected}**")

    # --- Configuration ---
    st.subheader("Configuration")
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
    with cfg_col1:
        chunking = st.selectbox("Chunking Strategy", CHUNKING_STRATEGIES, key="single_chunk")
    with cfg_col2:
        retrieval = st.selectbox("Retrieval Method", RETRIEVAL_METHODS, key="single_retr")
    with cfg_col3:
        agent = st.selectbox("Agent Architecture", AGENT_ARCHITECTURES, key="single_agent")
    with cfg_col4:
        model = st.selectbox("Model", MODELS, key="single_model")

    # --- Run ---
    if st.button("Fact-Check", type="primary", key="single_run"):
        if not claim_text or not claim_text.strip():
            st.warning("Please enter a claim first.")
        else:
            with st.spinner("Running pipeline..."):
                try:
                    result = run_pipeline(claim_text.strip(), chunking, retrieval, agent, model)
                    st.session_state["single_result"] = result
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    if "single_result" in st.session_state:
        st.divider()
        display_result(st.session_state["single_result"], header="Result")

# ============================== COMPARE ====================================
with tab_compare:
    st.header("Side-by-Side Comparison")

    # --- Claim input ---
    cmp_example = st.selectbox(
        "Select a curated example or type your own:",
        example_options,
        key="cmp_example",
    )

    if cmp_example == "(type your own)":
        cmp_claim = st.text_area(
            "Enter a health claim:",
            height=80,
            placeholder="e.g., Vitamin D supplements prevent COVID infection",
            key="cmp_claim_input",
        )
    else:
        idx = example_options.index(cmp_example) - 1
        cmp_claim = test_claims[idx]["claim"]
        expected_cmp = test_claims[idx].get("expected_verdict", "")
        st.text_area("Claim:", value=cmp_claim, height=80, disabled=True, key="cmp_claim_show")
        if expected_cmp:
            st.caption(f"Expected verdict: **{expected_cmp}**")

    # --- Two configurations ---
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Configuration A")
        a_chunk = st.selectbox("Chunking Strategy", CHUNKING_STRATEGIES, key="a_chunk")
        a_retr = st.selectbox("Retrieval Method", RETRIEVAL_METHODS, key="a_retr")
        a_agent = st.selectbox("Agent Architecture", AGENT_ARCHITECTURES, key="a_agent")
        a_model = st.selectbox("Model", MODELS, key="a_model")

    with right_col:
        st.subheader("Configuration B")
        b_chunk = st.selectbox("Chunking Strategy", CHUNKING_STRATEGIES, index=1, key="b_chunk")
        b_retr = st.selectbox("Retrieval Method", RETRIEVAL_METHODS, index=2, key="b_retr")
        b_agent = st.selectbox("Agent Architecture", AGENT_ARCHITECTURES, key="b_agent")
        b_model = st.selectbox("Model", MODELS, key="b_model")

    if st.button("Compare", type="primary", key="cmp_run"):
        if not cmp_claim or not cmp_claim.strip():
            st.warning("Please enter a claim first.")
        else:
            result_a = None
            result_b = None

            col_left, col_right = st.columns(2)

            with col_left:
                with st.spinner("Running Config A..."):
                    try:
                        result_a = run_pipeline(cmp_claim.strip(), a_chunk, a_retr, a_agent, a_model)
                    except Exception as e:
                        st.error(f"Config A error: {e}")

            with col_right:
                with st.spinner("Running Config B..."):
                    try:
                        result_b = run_pipeline(cmp_claim.strip(), b_chunk, b_retr, b_agent, b_model)
                    except Exception as e:
                        st.error(f"Config B error: {e}")

            st.session_state["cmp_result_a"] = result_a
            st.session_state["cmp_result_b"] = result_b

    if st.session_state.get("cmp_result_a") or st.session_state.get("cmp_result_b"):
        st.divider()
        res_left, res_right = st.columns(2)

        with res_left:
            if st.session_state.get("cmp_result_a"):
                display_result(st.session_state["cmp_result_a"], header="Config A Result")

        with res_right:
            if st.session_state.get("cmp_result_b"):
                display_result(st.session_state["cmp_result_b"], header="Config B Result")

        # Quick comparison summary
        a = st.session_state.get("cmp_result_a")
        b = st.session_state.get("cmp_result_b")
        if a and b:
            st.divider()
            st.subheader("Comparison Summary")
            summary_cols = st.columns(3)
            a_meta = a.get("metadata", {})
            b_meta = b.get("metadata", {})

            with summary_cols[0]:
                st.markdown("**Verdict Match**")
                match = a.get("verdict") == b.get("verdict")
                st.write("Yes" if match else "No")

            with summary_cols[1]:
                a_lat = a_meta.get("latency_seconds", 0)
                b_lat = b_meta.get("latency_seconds", 0)
                faster = "A" if a_lat <= b_lat else "B"
                st.markdown("**Faster**")
                st.write(f"Config {faster} ({min(a_lat, b_lat):.2f}s vs {max(a_lat, b_lat):.2f}s)")

            with summary_cols[2]:
                a_cost = a_meta.get("estimated_cost_usd", 0)
                b_cost = b_meta.get("estimated_cost_usd", 0)
                cheaper = "A" if a_cost <= b_cost else "B"
                st.markdown("**Cheaper**")
                st.write(f"Config {cheaper} (${min(a_cost, b_cost):.6f} vs ${max(a_cost, b_cost):.6f})")

# ============================= DASHBOARD ===================================
with tab_dashboard:
    st.header("Aggregate Results Dashboard")

    experiment_results = load_experiment_results()
    experiment_configs = load_experiment_configs()

    if not experiment_results:
        st.info(
            "No experiment results found. Run experiments first to populate "
            f"`{RESULTS_DIR}` with E*.json files."
        )
    else:
        # ---- Overview table ----
        st.subheader("Experiment Overview")

        overview_rows = []
        for eid, data in sorted(experiment_results.items()):
            config = data.get("config", {})
            total = data.get("total_claims", 0)
            correct = data.get("correct", 0)
            accuracy = data.get("accuracy", 0)
            overview_rows.append({
                "Experiment": eid,
                "Name": config.get("name", ""),
                "Chunking": config.get("chunking_strategy", ""),
                "Retrieval": config.get("retrieval_method", ""),
                "Agent": config.get("agent_architecture", ""),
                "Model": config.get("model", ""),
                "Claims": total,
                "Correct": correct,
                "Accuracy": f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) else str(accuracy),
            })

        st.dataframe(overview_rows, use_container_width=True, hide_index=True)

        # ---- Accuracy bar chart ----
        st.subheader("Accuracy by Experiment")
        chart_data = {
            row["Experiment"]: float(row["Accuracy"].strip("%")) / 100
            if isinstance(row["Accuracy"], str) and "%" in row["Accuracy"]
            else 0
            for row in overview_rows
        }
        st.bar_chart(chart_data)

        # ---- Per-experiment drill-down ----
        st.subheader("Experiment Details")
        selected_exp = st.selectbox(
            "Select experiment to inspect:",
            list(experiment_results.keys()),
            key="dash_exp_select",
        )

        if selected_exp:
            exp_data = experiment_results[selected_exp]
            exp_results = exp_data.get("results", [])

            if not exp_results:
                st.info("No individual results recorded for this experiment.")
            else:
                # Verdict distribution
                st.markdown("**Verdict Distribution**")
                verdict_counts: dict[str, int] = {}
                for r in exp_results:
                    v = r.get("verdict", "N/A")
                    verdict_counts[v] = verdict_counts.get(v, 0) + 1
                st.bar_chart(verdict_counts)

                # Aggregate metrics
                latencies = [
                    r.get("metadata", {}).get("latency_seconds", 0)
                    for r in exp_results
                    if r.get("metadata")
                ]
                tokens = [
                    r.get("metadata", {}).get("total_tokens", 0)
                    for r in exp_results
                    if r.get("metadata")
                ]
                costs = [
                    r.get("metadata", {}).get("estimated_cost_usd", 0)
                    for r in exp_results
                    if r.get("metadata")
                ]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Claims", len(exp_results))
                if latencies:
                    m2.metric("Avg Latency", f"{sum(latencies)/len(latencies):.2f} s")
                if tokens:
                    m3.metric("Avg Tokens", f"{sum(tokens)//len(tokens)}")
                if costs:
                    m4.metric("Total Cost", f"${sum(costs):.4f}")

                # Per-claim results table
                st.markdown("**Per-Claim Results**")
                claim_rows = []
                for r in exp_results:
                    meta = r.get("metadata", {})
                    claim_rows.append({
                        "Claim": r.get("claim", "")[:80],
                        "Verdict": r.get("verdict", ""),
                        "Expected": r.get("expected_verdict", ""),
                        "Correct": r.get("correct", False),
                        "Latency (s)": meta.get("latency_seconds", ""),
                        "Tokens": meta.get("total_tokens", ""),
                    })
                st.dataframe(claim_rows, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Sidebar: about / experiment presets
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "This app lets you fact-check health claims using configurable "
        "RAG pipelines with different chunking strategies, retrieval "
        "methods, agent architectures, and LLMs."
    )
    st.markdown("---")

    st.subheader("Quick Presets")
    exp_configs = load_experiment_configs()
    if exp_configs:
        preset = st.selectbox(
            "Load experiment preset:",
            ["(none)"] + list(exp_configs.keys()),
            format_func=lambda k: f"{k}: {exp_configs[k]['name']}" if k in exp_configs else k,
            key="sidebar_preset",
        )
        if preset != "(none)" and preset in exp_configs:
            cfg = exp_configs[preset]
            st.code(json.dumps(cfg, indent=2), language="json")
    else:
        st.caption("Experiment configs not available (import failed).")

    st.markdown("---")
    st.subheader("Verdict Taxonomy")
    for v, c in VERDICT_COLORS.items():
        st.markdown(
            f'<span style="color:{c};font-weight:bold;">{v}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Not medical advice. Consult a healthcare professional.")
