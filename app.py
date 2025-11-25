import os
from datetime import datetime

import streamlit as st
from groq import Groq

# -------------------------
# CONFIG & CLIENT
# -------------------------

st.set_page_config(
    page_title="Vehicle Insurance Claim Intelligence Console",
    layout="wide",
)

# -------------------------
# SIMPLE THEME TOGGLE + CSS
# -------------------------


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning("Set GROQ_API_KEY as an environment variable or in Streamlit secrets.")
client = Groq(api_key=GROQ_API_KEY)

# ... rest of your code EXACTLY as you pasted (agents, pipeline, UI, main()) ...
# make sure the rest of the code is unchanged below this line

def call_groq(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    msg = resp.choices[0].message
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        text_parts = []
        for part in content:
            t = getattr(part, "text", "")
            if t:
                text_parts.append(t)
        content = "".join(text_parts)
    return str(content).strip()


# -------------------------
# AGENTS
# -------------------------

def triage_agent_groq(claim_type: str, description: str) -> str:
    system = "You are a precise vehicle insurance triage engine."
    user = f"""
You will triage a motor insurance claim.

Claim type: {claim_type}
Customer incident description:
\"\"\"{description}\"\"\"

Tasks:
1. Classify:
   - severity: one of ["Low", "Medium", "High"]
   - priority: one of ["P1 – Urgent", "P2 – Standard", "P3 – Low"]
   - expected_cycle_time: short phrase (e.g. "1–2 days", "3–5 days").

2. Explain your reasoning in 2–3 lines, in simple language.

Return the answer as concise markdown bullet points.
"""
    return call_groq(system, user)


def damage_agent_groq(claim_type: str, car_model: str, description: str, has_photos: bool) -> str:
    system = "You are an auto damage estimator for Indian private passenger vehicles."
    user = f"""
Estimate likely vehicle damage for this claim.

Claim type: {claim_type}
Car make & model (with year if given): {car_model}
Photos attached: {"Yes" if has_photos else "No"}
Customer description:
\"\"\"{description}\"\"\"

Instructions:
- Consider that repair costs vary by segment:
  - Entry/budget cars (e.g. Alto, WagonR, Tiago) → cheaper parts & labour
  - Mid-range (e.g. Baleno, i20, City) → moderate
  - Premium/luxury (e.g. BMW, Mercedes, Audi) → high repair cost
- Use this to pick a realistic repair cost band.

Return:
- likely damaged parts
- type of damage (scratches, dents, broken components, glass damage, etc.)
- rough repair cost range in INR (e.g. "₹10,000–₹15,000")
- damage category: one of ["Minor", "Moderate", "Severe"]
- short notes (mention if photos would help confirm or refine estimate).

Return a short structured markdown answer.
"""
    return call_groq(system, user)


def evidence_agent_groq(description: str, has_photos: bool) -> str:
    system = "You are an insurance claims evidence and documentation assistant."
    user = f"""
You will help a claims adjuster quickly understand the case and decide what to ask the customer.

Customer's description of the accident:
\"\"\"{description}\"\"\"

Customer has already attached some photos: {"Yes" if has_photos else "No"}

Return:
1. A 3–4 line summary suitable for an internal adjuster.
2. A bullet list of MOST important missing information / documents to request next, such as:
   - more photos (mention which angles if needed)
   - police report / FIR / challan
   - workshop estimate
   - injury details
   - any other key info.

Return in markdown with headings:
- Summary
- Missing information / documents
"""
    return call_groq(system, user)


def settlement_agent_groq(claim_type: str, car_model: str, description: str) -> str:
    system = "You are a senior, fair motor insurance claims adjuster in India."
    user = f"""
You must suggest an initial settlement approach for this motor claim.

Claim type: {claim_type}
Car make & model: {car_model}
Customer description:
\"\"\"{description}\"\"\"

Assume:
- Private car comprehensive policy
- Usual deductibles apply
- Standard Indian repair market

Consider car segment:
- For budget cars, typical repairs are cheaper.
- For premium/luxury cars, parts & labour are significantly more expensive.

Return:
- Is this suitable for fast-track / straight-through processing? (yes/no + why)
- A reasonable settlement RANGE in INR (e.g. "₹15,000–₹20,000")
- 2–3 key decision factors you used
- Any risks or caveats (e.g. missing documents, unclear liability, potential fraud).

Return in clear markdown bullet points, no legal jargon.
"""
    return call_groq(system, user)


def customer_message_agent_groq(claim_id: str, description: str, evidence_markdown: str) -> str:
    system = "You draft short, polite messages for insurance policyholders."
    user = f"""
You are helping a motor insurance company talk to a customer.

Claim ID: {claim_id}

Customer's original description of the accident:
\"\"\"{description}\"\"\"

Internal notes from the Evidence Agent (includes missing information section):
\"\"\"{evidence_markdown}\"\"\"

Write a short message that:
- thanks the customer
- clearly lists what documents/photos/details they need to share
- is in simple, friendly English
- is suitable for WhatsApp or email
- is under 120 words.

Do NOT include headings or technical language.
Only return the message body.
"""
    return call_groq(system, user)


# -------------------------
# PIPELINE & UTILS
# -------------------------

def extract_priority_label(triage_md: str) -> str:
    txt = triage_md.lower()
    if "high" in txt:
        return "HIGH"
    if "medium" in txt:
        return "MEDIUM"
    if "low" in txt:
        return "LOW"
    return "UNSPECIFIED"


def run_claim_pipeline_groq(
    claim_id: str,
    claim_type: str,
    channel: str,
    car_model: str,
    description: str,
    has_photos: bool,
):
    triage = triage_agent_groq(claim_type, description)
    damage = damage_agent_groq(claim_type, car_model, description, has_photos)
    evidence = evidence_agent_groq(description, has_photos)
    settlement = settlement_agent_groq(claim_type, car_model, description)

    overview = {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "channel": channel,
        "car_model": car_model,
        "has_photos": has_photos,
    }

    return {
        "overview": overview,
        "triage": triage,
        "damage": damage,
        "evidence": evidence,
        "settlement": settlement,
    }


def init_state():
    if "claims_db" not in st.session_state:
        st.session_state.claims_db = {}  # claim_id -> record
    if "active_claim_id" not in st.session_state:
        st.session_state.active_claim_id = None
    if "next_idx" not in st.session_state:
        st.session_state.next_idx = 0
    if "customer_msg" not in st.session_state:
        st.session_state.customer_msg = ""
    if "send_status" not in st.session_state:
        st.session_state.send_status = ""


def save_claim_record(
    claim_id,
    customer_name,
    customer_contact,
    claim_type,
    channel,
    car_model,
    description,
    pipeline_result,
    status="Reviewing",
):
    st.session_state.next_idx += 1
    triage_md = pipeline_result["triage"]
    priority = extract_priority_label(triage_md)

    st.session_state.claims_db[claim_id] = {
        "id": claim_id,
        "customer_name": customer_name,
        "customer_contact": customer_contact,
        "type": claim_type,
        "channel": channel,
        "car_model": car_model,
        "description": description,
        "status": status,
        "priority": priority,
        "triage": triage_md,
        "damage": pipeline_result["damage"],
        "evidence": pipeline_result["evidence"],
        "settlement": pipeline_result["settlement"],
        "overview": pipeline_result["overview"],
        "created_at": datetime.now(),
        "created_idx": st.session_state.next_idx,
    }


def format_time(dt: datetime) -> str:
    return dt.strftime("%d %b %Y · %H:%M")


def get_sorted_claims():
    return sorted(
        st.session_state.claims_db.values(),
        key=lambda c: c["created_idx"],
        reverse=True,
    )


# -------------------------
# UI FUNCTIONS
# -------------------------

def render_priority_chip(priority: str):
    priority = (priority or "UNSPECIFIED").upper()
    if priority == "HIGH":
        color = "#fee2e2"
        text_color = "#b91c1c"
    elif priority == "MEDIUM":
        color = "#fef3c7"
        text_color = "#92400e"
    elif priority == "LOW":
        color = "#dcfce7"
        text_color = "#166534"
    else:
        color = "#e5e7eb"
        text_color = "#4b5563"

    html = f"""
    <span style="
        background:{color};
        color:{text_color};
        padding:2px 10px;
        border-radius:999px;
        font-size:11px;
        font-weight:600;
        letter-spacing:0.03em;
    ">{priority}</span>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_history(priority_filter: str):
    claims = get_sorted_claims()
    if priority_filter != "All":
        claims = [c for c in claims if c["priority"] == priority_filter.upper()]

    if not claims:
        st.write("_No claims for this filter._")
        return

    # 2-column grid of compact cards
    cols = st.columns(2)

    for idx, claim in enumerate(claims):
        cid = claim["id"]
        name = claim["customer_name"]
        created_at = claim["created_at"]
        priority = claim["priority"]
        status = claim["status"]
        chan = claim["channel"]
        contact = claim["customer_contact"]

        col = cols[idx % 2]  # left / right alternation

        with col:
            # Card header + meta (top part)
            st.markdown(
                f"""
                <div class="claim-card">
                    <div class="claim-card-header">#{cid} · {name}</div>
                    <div class="claim-card-meta">
                        {format_time(created_at)} · {chan}{(" · " + contact) if contact else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Single row: priority · status · delete · open
            c1, c2, c3, c4 = st.columns([2, 3, 1, 2])

            with c1:
                render_priority_chip(priority)

            with c2:
                if status == "Completed":
                    st.markdown("🟢 Completed")
                else:
                    st.markdown("🟠 Reviewing")

            with c3:
                delete_clicked = st.button(
                    "🗑️",
                    key=f"del_{cid}",
                    help="Delete claim",
                )

            with c4:
                open_clicked = st.button(
                    "🔍 Open",
                    key=f"open_{cid}",
                    help="Open in review panel",
                )

            # Actions
            if open_clicked:
                st.session_state.active_claim_id = cid
                st.session_state.customer_msg = ""
                st.session_state.send_status = ""
                st.rerun()

            if delete_clicked:
                del st.session_state.claims_db[cid]
                st.session_state.send_status = ""
                st.session_state.customer_msg = ""
                st.rerun()



def render_claim_intelligence():
    cid = st.session_state.active_claim_id
    if not cid or cid not in st.session_state.claims_db:
        st.info("No active claim selected. File a new claim or select one from History.")
        return

    claim = st.session_state.claims_db[cid]
    ov = claim["overview"]

    st.subheader("🧠 Claim Intelligence Review")

    st.markdown(
        f"**Claim ID:** {ov['claim_id']}  \n"
        f"**Customer:** {claim['customer_name']}  \n"
        f"**Channel:** {ov['channel']}  \n"
        f"**Claim type:** {ov['claim_type']}  \n"
        f"**Vehicle:** {ov['car_model']}  \n"
        f"**Photos attached:** {'Yes' if ov['has_photos'] else 'No'}"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🚦 Triage Assessment", "🔧 Damage & Repair", "📑 Evidence & Docs", "💰 Settlement Guidance"]
    )

    with tab1:
        st.markdown(claim["triage"])

    with tab2:
        st.markdown(claim["damage"])

    with tab3:
        st.markdown(claim["evidence"])
        st.markdown("---")
        st.markdown("#### 📩 Request Missing Information")
        if st.button("Generate customer message"):
            st.session_state.customer_msg = customer_message_agent_groq(
                claim_id=cid,
                description=claim["description"],
                evidence_markdown=claim["evidence"],
            )
            st.session_state.send_status = ""
        if st.session_state.customer_msg:
            st.text_area(
                "Customer communication draft",
                st.session_state.customer_msg,
                height=120,
            )
            if st.button("Simulate send message"):
                st.session_state.send_status = "✅ Message marked as sent to customer (simulated)."
        if st.session_state.send_status:
            st.success(st.session_state.send_status)

    with tab4:
        st.markdown(claim["settlement"])

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save as Reviewing"):
            claim["status"] = "Reviewing"
            st.session_state.active_claim_id = None
            st.session_state.customer_msg = ""
            st.session_state.send_status = ""
            st.success(f"Claim {cid} saved as Reviewing.")
            st.rerun()
    with col2:
        if st.button("✅ Mark as Completed"):
            claim["status"] = "Completed"
            st.session_state.active_claim_id = None
            st.session_state.customer_msg = ""
            st.session_state.send_status = ""
            st.success(f"Claim {cid} marked as Completed.")
            st.rerun()
    with col3:
        if st.button("🆕 Start a New Claim"):
            st.session_state.active_claim_id = None
            st.session_state.customer_msg = ""
            st.session_state.send_status = ""
            st.rerun()

def inject_global_css():
    css = """
    <style>
    .claim-card {
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 10px;
        background: #111827; /* Dark slate background */
        border: 1px solid rgba(255,255,255,0.08);
    }
    .claim-card-header {
        font-weight: 600;
        font-size: 15px;
        color: white;
    }
    .claim-card-meta {
        font-size: 11px;
        opacity: 0.8;
        color: #d1d5db;
        margin-top: 2px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)



# -------------------------
# MAIN APP
# -------------------------

def main():
    init_state()
    inject_global_css() 

    st.title("Vehicle Insurance Claim Intelligence Console")
    st.markdown(
        "### AI-assisted workspace for motor claim adjusters\n"
        "Capture FNOL details, run multi-agent analysis, "
        "and manage claims through an AI-prioritized inbox."
    )

    tab_ws, tab_hist = st.tabs(["Workspace", "History"])

    with tab_ws:
        # if no active claim → show form
        if st.session_state.active_claim_id is None:
            st.subheader("📝 File a New Motor Claim")

            with st.form("new_claim_form"):
                col1, col2 = st.columns(2)
                with col1:
                    claim_id = st.text_input("Claim ID", value=f"CLM-{st.session_state.next_idx + 1:03d}")
                    customer_name = st.text_input("Customer name")
                    customer_contact = st.text_input(
                        "Customer contact (email or phone)",
                        placeholder="e.g. rahul.mehta@example.com / +91-98XXXXXXX",
                    )
                    claim_type = st.selectbox(
                        "Claim type",
                        ["Collision", "Rear-end", "Windscreen", "Parking damage"],
                    )
                with col2:
                    channel = st.selectbox(
                        "Intake channel",
                        ["Mobile App", "Call Centre", "Web Portal"],
                    )
                    car_model = st.text_input(
                        "Vehicle make & model",
                        value="Maruti Suzuki Baleno 2022",
                    )
                    description = st.text_area(
                        "Claim description (FNOL text)",
                        height=140,
                        placeholder=(
                            "Another car hit my front bumper in slow traffic near a signal. "
                            "The front bumper and left headlight are damaged. No one was injured."
                        ),
                    )
                    photo = st.file_uploader("Upload accident photo (optional)", type=["jpg", "jpeg", "png"])

                submitted = st.form_submit_button("▶ Analyze Claim")

            if submitted:
                if not description.strip() or not customer_name.strip():
                    st.error("Please fill at least customer name and description.")
                else:
                    photo_present = photo is not None
                    with st.spinner("Running multi-agent analysis..."):
                        pipeline_result = run_claim_pipeline_groq(
                            claim_id=claim_id,
                            claim_type=claim_type,
                            channel=channel,
                            car_model=car_model,
                            description=description,
                            has_photos=photo_present,
                        )
                    save_claim_record(
                        claim_id=claim_id,
                        customer_name=customer_name,
                        customer_contact=customer_contact,
                        claim_type=claim_type,
                        channel=channel,
                        car_model=car_model,
                        description=description,
                        pipeline_result=pipeline_result,
                        status="Reviewing",
                    )
                    st.session_state.active_claim_id = claim_id
                    st.session_state.customer_msg = ""
                    st.session_state.send_status = ""
                    st.success(f"Analysis completed for claim {claim_id}. Scroll down to view details.")
                    st.rerun()
        else:
            render_claim_intelligence()

    with tab_hist:
        st.subheader("📚 Claim Records")
        st.caption("Prioritized inbox of all claims. Filter by priority and click a row to open or delete.")

        priority_filter = st.selectbox("Filter by priority", ["All", "High", "Medium", "Low"])
        render_history(priority_filter)


if __name__ == "__main__":
    main()


# ---------------------------
# STREAMLIT APP ENDS HERE
# ---------------------------
