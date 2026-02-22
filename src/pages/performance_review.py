"""Performance Review page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note, _definition_block,
    _takeaway_block, _page_conclusion, _page_footer, _add_boj_events,
    _about_page_styles, _PALETTE,
)
from src.pages._data import load_unified, load_rates, load_market, _safe_col
from src.pages.regime import _run_ensemble, _run_markov, _run_entropy, _run_garch
from src.pages.early_warning import _run_warning_score, _run_ml_predictor
from src.pages.yield_curve import _run_pca


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")

from datetime import datetime
from src.reporting.metrics_tracker import AccuracyTracker, generate_improvement_suggestions
from src.reporting.pdf_export import JGBReportPDF, dataframe_to_csv_bytes



def page_performance_review():
    st.header("Performance Review")
    _page_intro(
        "Model performance tracking, accuracy metrics, and improvement suggestions. "
        "This page evaluates the regime detection ensemble against actual market outcomes "
        "and provides actionable recommendations for model refinement. Export reports as PDF or CSV."
    )
    _definition_block(
        "How Performance is Measured",
        "The framework tracks every regime prediction the ensemble makes and compares it against "
        "what actually happened. <b>Accuracy</b> is the percentage of correct predictions. "
        "<b>Precision</b> measures how often a repricing signal is genuine (vs false alarm). "
        "<b>Recall</b> measures how many actual repricing events the model caught. "
        "<b>False Positive Rate</b> is how often the model cried wolf. "
        "A good model has high precision (few false alarms) AND high recall (catches most real events). "
        "The improvement suggestions below are generated automatically based on which metrics need attention."
    )

    args = _get_args()

    # --- Compute metrics from ensemble predictions ---
    tracker = AccuracyTracker()
    ensemble = None
    predicted = None
    actual = None
    df = None

    try:
        ensemble = _run_ensemble(*args)
        df = load_unified(*_get_args())
        if ensemble is not None and len(ensemble.dropna()) > 30:
            ens_clean = ensemble.dropna()
            predicted = (ens_clean > 0.5).astype(int)
            jp10 = df["JP_10Y"].dropna() if "JP_10Y" in df.columns else pd.Series(dtype=float)
            if len(jp10) > 30:
                future_change = jp10.diff(5).shift(-5)
                vol = jp10.diff().rolling(60).std()
                actual = ((future_change > vol).astype(int)).reindex(predicted.index)
                metrics = tracker.compute_from_series(predicted, actual.dropna())
            else:
                metrics = tracker.compute_metrics()
        else:
            metrics = tracker.compute_metrics()
    except Exception as exc:
        st.warning(f"Could not compute metrics from ensemble: {exc}")
        metrics = tracker.compute_metrics()

    # --- KPI row ---
    try:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics.prediction_accuracy:.1%}")
        c2.metric("Precision", f"{metrics.precision:.1%}")
        c3.metric("Recall", f"{metrics.recall:.1%}")
        c4.metric("False Positive Rate", f"{metrics.false_positive_rate:.1%}")
    except Exception as exc:
        st.warning(f"Could not render KPI metrics: {exc}")

    # --- Detailed metrics table ---
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "False Positive Rate",
                    "Average Lead Time", "Total Predictions"],
        "Value": [f"{metrics.prediction_accuracy:.1%}", f"{metrics.precision:.1%}",
                  f"{metrics.recall:.1%}", f"{metrics.false_positive_rate:.1%}",
                  f"{metrics.average_lead_time:.1f} days", str(metrics.total_predictions)],
    })
    try:
        st.subheader("Detailed Metrics")
        _section_note(
            f"Based on {metrics.total_predictions} observations. "
            f"Average lead time: {metrics.average_lead_time:.1f} days."
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Could not render detailed metrics: {exc}")

    # --- Confusion matrix visualization ---
    try:
        if predicted is not None and actual is not None:
            st.subheader("Prediction Analysis")
            _section_note(
                "Comparison of predicted regime states vs actual outcomes. "
                "The ensemble predicts repricing when probability > 0.5."
            )
            # Align predicted and actual
            common_idx = predicted.dropna().index.intersection(actual.dropna().index)
            if len(common_idx) > 10:
                p_aligned = predicted.loc[common_idx]
                a_aligned = actual.loc[common_idx]
                tp = int(((p_aligned == 1) & (a_aligned == 1)).sum())
                fp = int(((p_aligned == 1) & (a_aligned == 0)).sum())
                fn = int(((p_aligned == 0) & (a_aligned == 1)).sum())
                tn = int(((p_aligned == 0) & (a_aligned == 0)).sum())
                cm1, cm2, cm3, cm4 = st.columns(4)
                cm1.metric("True Positives", f"{tp}")
                cm2.metric("False Positives", f"{fp}")
                cm3.metric("True Negatives", f"{tn}")
                cm4.metric("False Negatives", f"{fn}")

                # Ensemble probability over time with actual regime overlay
                if ensemble is not None:
                    ens_clean = ensemble.dropna()
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=ens_clean.index, y=ens_clean.values,
                        mode="lines", name="Ensemble Probability",
                        line=dict(color="#CFB991", width=2),
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=a_aligned.index, y=a_aligned.values * 0.95,
                        mode="markers", name="Actual Repricing",
                        marker=dict(color="#c0392b", size=4, opacity=0.5),
                    ))
                    fig_pred.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Decision Boundary")
                    fig_pred.update_layout(yaxis_title="Probability / Actual", yaxis_range=[0, 1])
                    _add_boj_events(fig_pred)
                    _chart(_style_fig(fig_pred, 380))

                    _takeaway_block(
                        f"Out of {len(common_idx)} aligned observations: <b>{tp} true positives</b> (correctly predicted repricing), "
                        f"<b>{fp} false positives</b> (false alarms), <b>{fn} false negatives</b> (missed repricing events), "
                        f"and <b>{tn} true negatives</b> (correctly predicted suppression)."
                    )
    except Exception as exc:
        st.warning(f"Could not render prediction analysis: {exc}")

    # --- Improvement suggestions ---
    suggestions = generate_improvement_suggestions(metrics)
    try:
        st.subheader("Improvement Suggestions")
        if suggestions:
            for s in suggestions:
                st.info(s)
        else:
            st.success("All metrics are within acceptable ranges. No immediate improvements needed.")
    except Exception as exc:
        st.warning(f"Could not render improvement suggestions: {exc}")

    # --- Ensemble model agreement analysis ---
    try:
        st.subheader("Model Agreement Analysis")
        _section_note(
            "How often do the four regime detection models agree? Higher agreement means stronger conviction."
        )
        markov_result = _run_markov(*args)
        _, ent_sig = _run_entropy(*args)
        vol, _ = _run_garch(*args)

        model_signals = {}
        if ensemble is not None and len(ensemble.dropna()) > 0:
            model_signals["Ensemble"] = (ensemble.dropna() > 0.5).astype(int)
        # markov_result is a dict with 'regime_probabilities' key
        if markov_result is not None and isinstance(markov_result, dict):
            markov_prob = markov_result.get("regime_probabilities")
            if markov_prob is not None and len(markov_prob) > 0:
                prob_col = markov_prob.columns[-1]
                mp = markov_prob[prob_col].dropna()
                if len(mp) > 0:
                    model_signals["Markov"] = (mp > 0.5).astype(int)
        if ent_sig is not None and hasattr(ent_sig, "dropna") and len(ent_sig.dropna()) > 0:
            model_signals["Entropy"] = ent_sig.dropna().astype(int)
        # vol is a Series from GARCH
        if vol is not None and hasattr(vol, "dropna") and len(vol.dropna()) > 0:
            vol_clean = vol.dropna()
            vol_median = vol_clean.median()
            model_signals["GARCH"] = (vol_clean > vol_median).astype(int)

        if len(model_signals) >= 2:
            agreement_df = pd.DataFrame(model_signals)
            agreement_df = agreement_df.dropna()
            if len(agreement_df) > 0:
                agreement_rate = agreement_df.apply(lambda row: row.nunique() == 1, axis=1).mean()
                st.metric("Model Agreement Rate", f"{agreement_rate:.1%}")
                _takeaway_block(
                    f"The regime detection models agree <b>{agreement_rate:.0%}</b> of the time. "
                    f"{'High agreement strengthens signal conviction.' if agreement_rate > 0.7 else 'Moderate agreement suggests uncertainty in regime classification.' if agreement_rate > 0.5 else 'Low agreement indicates conflicting signals. Rely on the ensemble average rather than any single model.'}"
                )
        else:
            st.info("Insufficient model outputs to compute agreement analysis.")
    except Exception as exc:
        st.warning(f"Could not render model agreement analysis: {exc}")

    # --- ML Regime Predictor (Enhancement 1b) ---
    if _get_layout_config().show_ml_predictions:
        try:
            st.subheader("ML Regime Predictor")
            _definition_block(
                "RandomForest Walk-Forward Predictor",
                "A machine learning model trained on features derived from the framework's analytics. "
                "Uses walk-forward training (504-day window, retrained quarterly) to avoid look-ahead bias. "
                "Features include structural entropy, carry stress, spillover correlation, volatility z-score, "
                "and USDJPY momentum."
            )
            with st.spinner("Running ML predictor..."):
                preds, probs, importance = _run_ml_predictor(
                    *_get_args(), _get_layout_config().entropy_window,
                )

            if probs is not None and len(probs.dropna()) > 0:
                latest_prob = float(probs.dropna().iloc[-1])
                _section_note(
                    f"ML prediction probability (latest): <b>{latest_prob:.0%}</b>. "
                    f"Walk-forward trained on {len(probs.dropna())} prediction points."
                )
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(
                    x=probs.index, y=probs.values,
                    mode="lines", name="ML Regime Prob",
                    line=dict(color="#CFB991", width=2),
                ))
                fig_ml.add_hline(y=0.5, line_dash="dash", line_color="red")
                fig_ml.update_layout(yaxis_title="Probability", yaxis_range=[0, 1])
                _add_boj_events(fig_ml)
                _chart(_style_fig(fig_ml, 380))

                _takeaway_block(
                    f"The ML predictor assigns a <b>{latest_prob:.0%}</b> probability to the repricing regime. "
                    f"{'This confirms the ensemble signal. Both statistical and ML models agree.' if (latest_prob > 0.5) == (metrics.prediction_accuracy > 0.5) else 'The ML model disagrees with the ensemble. When models diverge, reduce position sizes and wait for convergence.'}"
                )

                # Feature importance
                if importance is not None and len(importance) > 0:
                    st.subheader("Feature Importance")
                    _section_note("RandomForest feature importances from the latest training window.")
                    fig_imp = go.Figure(go.Bar(
                        x=importance.values,
                        y=importance.index,
                        orientation="h",
                        marker_color="#CFB991",
                    ))
                    fig_imp.update_layout(xaxis_title="Importance", yaxis_title="Feature")
                    _chart(_style_fig(fig_imp, 320))

                    top_feature = importance.idxmax() if len(importance) > 0 else "N/A"
                    _takeaway_block(
                        f"The most important feature is <b>{top_feature}</b>. "
                        "Feature importances reveal which market signals the ML model relies on most heavily. "
                        "If a single feature dominates (>40%), the model may be fragile to changes in that signal."
                    )
            else:
                st.info("Insufficient data for ML regime predictor. Need at least 100 observations with both regime classes present in the training window.")
        except ImportError:
            st.info("ML predictor requires scikit-learn. Install with: `pip install scikit-learn`")
        except Exception as exc:
            st.warning(f"ML predictor not available: {exc}")

    # --- Export ---
    try:
        st.subheader("Export Reports")
        _section_note(
            "Download profile-tailored PDFs or raw metrics CSV. "
            "**Trader**: regime + trades. **Analyst**: full analysis. **Academic**: methodology + references."
        )

        _pr_args = _get_args()
        _pr_kwargs = dict(
            metrics=metrics,
            suggestions=suggestions,
        )
        # Gather ML context
        try:
            _, _pr_ml_p, _pr_ml_i = _run_ml_predictor(*_pr_args, _get_layout_config().entropy_window)
            if _pr_ml_p is not None and len(_pr_ml_p.dropna()) > 0:
                _pr_kwargs["ml_prob"] = float(_pr_ml_p.dropna().iloc[-1])
                _pr_kwargs["ml_importance"] = _pr_ml_i
        except Exception:
            pass
        try:
            _pr_ens = _run_ensemble(*_pr_args)
            if _pr_ens is not None and len(_pr_ens.dropna()) > 0:
                _pr_kwargs["ensemble_prob"] = float(_pr_ens.dropna().iloc[-1])
        except Exception:
            pass
        try:
            _pr_pca = _run_pca(*_pr_args)
            _pr_kwargs["pca_result"] = _pr_pca
        except Exception:
            pass
        try:
            _pr_ws = _run_warning_score(*_pr_args, _get_layout_config().entropy_window)
            if _pr_ws is not None and len(_pr_ws.dropna()) > 0:
                _pr_kwargs["warning_score"] = float(_pr_ws.dropna().iloc[-1])
        except Exception:
            pass

        col_t, col_a, col_ac, col_csv = st.columns(4)
        for _col, _prof, _key in [
            (col_t,  "Trader",   "pr_pdf_trader"),
            (col_a,  "Analyst",  "pr_pdf_analyst"),
            (col_ac, "Academic", "pr_pdf_academic"),
        ]:
            with _col:
                try:
                    _r = JGBReportPDF()
                    _r.add_title_page(
                        title=f"Performance Review  -  {_prof}",
                        subtitle=f"Generated {datetime.now():%Y-%m-%d %H:%M}",
                    )
                    _r.add_full_analysis_report(_prof, **_pr_kwargs)
                    st.download_button(
                        f"{_prof} PDF",
                        data=_r.to_bytes(),
                        file_name=f"jgb_review_{_prof.lower()}_{datetime.now():%Y%m%d}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key=_key,
                    )
                except Exception as exc:
                    st.warning(f"Could not generate {_prof} PDF: {exc}")

        with col_csv:
            try:
                csv_bytes = dataframe_to_csv_bytes(metrics_df)
                st.download_button(
                    "Metrics CSV",
                    data=csv_bytes,
                    file_name=f"jgb_metrics_{datetime.now():%Y%m%d}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception as exc:
                st.warning(f"Could not generate CSV: {exc}")
    except Exception as exc:
        st.warning(f"Could not render export section: {exc}")

    # --- Page conclusion (always render, even if sections above failed) ---
    try:
        _verdict_pr = (
            f"Accuracy at {metrics.prediction_accuracy:.0%}: "
            f"{'strong model performance' if metrics.prediction_accuracy > 0.7 else 'room for improvement' if metrics.prediction_accuracy > 0.5 else 'model needs retraining'}."
        )
        _page_conclusion(
            _verdict_pr,
            f"Performance metrics computed over {metrics.total_predictions} observations. "
            f"{len(suggestions)} improvement suggestion(s) generated.",
        )
    except Exception:
        pass
    _page_footer()


