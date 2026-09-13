import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.agent.pipeline import run_agent


st.set_page_config(
    page_title="Apple Support AI Agent",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Apple Support AI Agent")
st.write("AI-powered customer support assistant using historical AppleSupport cases.")

st.divider()

customer_message = st.text_area(
    "Customer Message",
    placeholder="Example: My iPhone battery is draining very fast after the latest update.",
    height=150
)

if st.button("🔍 Analyze Message", type="primary"):

    if not customer_message.strip():
        st.warning("Please enter a customer message.")
    else:
        with st.spinner("Analyzing customer message..."):
            result = run_agent(customer_message)

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Intent", result.intent)

        with col2:
            st.metric("Confidence", f"{result.confidence:.0%}")

        with col3:
            st.metric("Decision", result.decision)

        st.subheader("📝 AI Reply")
        st.info(result.reply)

        st.subheader("💡 Reason")
        st.write(result.reason)

        st.subheader("📚 Historical Evidence")

        if result.evidence:
            for i, evidence in enumerate(result.evidence, 1):
                st.write(f"**Case {i}:**")
                st.write(evidence)
                st.divider()
        else:
            st.write("No historical evidence found.")