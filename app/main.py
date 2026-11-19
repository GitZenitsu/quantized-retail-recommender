"""Streamlit entry point: `streamlit run app/main.py`"""
import streamlit as st

from app import core

st.set_page_config(page_title="Quantized Retail Recommender")
st.title("Quantized Retail Recommender")
st.write(
    "A smart retail recommender that identifies frequently bought items. It then shrinks "
    "the recommendation model using quantization to reduce its footprint for edge device "
    "deployment or faster loading."
)
st.caption("Trained on synthetic shopping baskets, so the suggestions show the method, not real shoppers.")


@st.cache_resource
def load_models():
    baseline = core._get_trained()[2]
    dynamic, static = core.milestone_5(baseline)
    return {"Baseline (float32)": baseline, "Dynamic int8": dynamic, "Static int8": static}


models = load_models()

st.subheader("1. What is in the basket?")
basket = st.multiselect(
    "Add items in the order they were picked (the last one drives the suggestion)",
    core.ITEMS,
    default=["bread"],
)

if basket:
    choice = st.selectbox("Model", list(models))
    st.subheader("2. Suggested next items")
    for item, prob in core.predict_next(models[choice], basket, k=3):
        st.write(f"**{item}**: {prob:.0%}")
else:
    st.info("Pick at least one item to see suggestions.")

st.subheader("3. Do these items go together?")
if len(basket) >= 2:
    st.write(f"Together score: **{core.itemset_score(models['Baseline (float32)'], basket):.2f}** (0 to 1)")
else:
    st.write("Pick two or more items to probe them as a set.")

st.subheader("4. Quantization results")
report = core.milestone_6()
rows = []
for name, key in (("Baseline (float32)", "baseline"), ("Dynamic int8", "dynamic"), ("Static int8", "static")):
    row = report[key]
    rows.append({
        "Model": name,
        "Size (bytes)": row["size_bytes"],
        "Top-1 accuracy": f"{row['top1']:.1%}",
        "Top-3 accuracy": f"{row['top3']:.1%}",
    })
st.table(rows)
