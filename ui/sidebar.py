"""
Sidebar rendering — alleen upload-widget.

Alle andere configuratie is verplaatst naar de hoofdtabs.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════
# Upload
# ═══════════════════════════════════════════════════════════════════════════

def render_upload_section() -> tuple[Optional[object], str]:
    """Toon upload-widget en input mode selector in de sidebar.

    Returns
    -------
    uploaded_file : UploadedFile | None
    input_mode : str
    """
    has_file = st.session_state.get("file_uploader_key") is not None

    st.sidebar.header("Upload je data _" if not has_file else "Ander bestand _")

    input_mode = st.sidebar.radio(
        "Kies het type bestand",
        ["Ruwe Aansluitingen (CSV)", "Reeds Geaggregeerd (CSV)"],
        key="input_mode",
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV", type=["csv"], key="file_uploader_key"
    )

    if not has_file:
        st.sidebar.info("Upload een CSV om te starten.")

    return uploaded_file, input_mode
