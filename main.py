# app.py
# Two-tab Streamlit demo using SQLite shared state + Admin panel + Event log export
# Run: python -m streamlit run app.py

import os
import json
import time
import random
import sqlite3
from datetime import datetime, timezone

import streamlit as st

# -----------------------------
# Parameters (demo-friendly)
# -----------------------------
C0 = 8.00
V0 = 14.00
P_SHOCK = 0.20
C_SHOCK_MULT = 1.10
V_SHOCK_MULT = 0.90

K1 = 1.00          # renegotiation cost per party if reneg triggered
K2 = 2.00          # escalated cost if you later add another reneg stage

W_MIN, W_MAX = 8.00, 14.00
DB_PATH = "tce_demo.sqlite3"

ADMIN_KEY = os.getenv("TCE_ADMIN_KEY", "admin")  # demo admin key

# -----------------------------
# DB helpers
# -----------------------------
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_code TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            supplier_pin TEXT NOT NULL,
            producer_pin TEXT NOT NULL,
            state_json TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_code TEXT NOT NULL,
            ts REAL NOT NULL,
            actor TEXT NOT NULL,           -- "Supplier" | "Producer" | "System" | "Admin"
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_code, ts);")

    conn.commit()
    conn.close()

def now() -> float:
    return time.time()

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def log_event(session_code: str, actor: str, event_type: str, payload: dict | None = None):
    payload = payload or {}
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events(session_code, ts, actor, event_type, payload_json) VALUES(?,?,?,?,?)",
        (session_code.upper(), now(), actor, event_type, json.dumps(payload))
    )
    conn.commit()
    conn.close()

def default_state():
    return {
        "phase": "R1_SUPPLIER_OFFER",
        "ended": False,
        "end_reason": None,

        "reneg_triggered": False,
        "k_paid_each": 0.0,

        "w1_offer": None,
        "w1_counter": None,
        "w1_final": None,
        "w1_agreed": None,

        "w2_offer": None,
        "w2_counter": None,
        "w2_final": None,
        "w2_agreed": None,

        "shocks_drawn": False,
        "supplier_cost_shock": None,
        "producer_demand_shock": None,
        "c_real": None,
        "v_real": None,

        "supplier_wants_reneg": None,
        "producer_wants_reneg": None,

        "w_final": None,
        "payoffs": None,

        "rng_seed": None,
    }

def load_session(session_code: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE session_code = ?", (session_code.upper(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    state = json.loads(row["state_json"])
    return {
        "session_code": row["session_code"],
        "created_at": row["created_at"],
        "supplier_pin": row["supplier_pin"],
        "producer_pin": row["producer_pin"],
        "state": state,
    }

def save_state(session_code: str, state: dict):
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE sessions SET state_json = ? WHERE session_code = ?",
        (json.dumps(state), session_code.upper())
    )
    conn.commit()
    conn.close()

def create_session(session_code: str, supplier_pin: str, producer_pin: str, rng_seed: int | None = None):
    session_code = session_code.upper().strip()
    state = default_state()
    state["rng_seed"] = rng_seed
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions(session_code, created_at, supplier_pin, producer_pin, state_json) VALUES(?,?,?,?,?)",
        (session_code, now(), supplier_pin, producer_pin, json.dumps(state))
    )
    conn.commit()
    conn.close()

    log_event(session_code, "Admin", "SESSION_CREATED", {"rng_seed": rng_seed})

def list_sessions():
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT session_code, created_at, state_json FROM sessions ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        state = json.loads(r["state_json"])
        out.append({
            "session_code": r["session_code"],
            "created_at": r["created_at"],
            "phase": state.get("phase"),
            "ended": bool(state.get("ended")),
            "reneg_triggered": bool(state.get("reneg_triggered")),
            "w_final": state.get("w_final"),
            "end_reason": state.get("end_reason"),
        })
    return out

def get_events(session_code: str):
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, ts, actor, event_type, payload_json FROM events WHERE session_code = ? ORDER BY ts ASC, id ASC",
        (session_code.upper(),)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def reset_session(session_code: str):
    sess = load_session(session_code)
    if not sess:
        return False
    state = default_state()
    # preserve seed if it existed (nice for reproducible demos)
    state["rng_seed"] = sess["state"].get("rng_seed")
    save_state(session_code, state)

    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM events WHERE session_code = ?", (session_code.upper(),))
    conn.commit()
    conn.close()

    log_event(session_code, "Admin", "SESSION_RESET", {})
    return True

def delete_session(session_code: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM events WHERE session_code = ?", (session_code.upper(),))
    cur.execute("DELETE FROM sessions WHERE session_code = ?", (session_code.upper(),))
    conn.commit()
    conn.close()

# -----------------------------
# Game logic helpers
# -----------------------------
def money(x: float) -> str:
    return f"${x:,.2f}"

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def phase_title(phase: str) -> str:
    return {
        "R1_SUPPLIER_OFFER": "Round 1 — Supplier proposes price",
        "R1_PRODUCER_RESPONSE": "Round 1 — Producer responds",
        "R1_SUPPLIER_FINAL": "Round 1 — Supplier final offer",
        "R1_PRODUCER_FINAL": "Round 1 — Producer accept/reject final",
        "SHOCKS": "Shocks realized (private)",
        "RENEG_DECISION": "Renegotiation decision",
        "R2_SUPPLIER_OFFER": "Round 2 — Supplier proposes price",
        "R2_PRODUCER_RESPONSE": "Round 2 — Producer responds",
        "R2_SUPPLIER_FINAL": "Round 2 — Supplier final offer",
        "R2_PRODUCER_FINAL": "Round 2 — Producer accept/reject final",
        "RESULTS": "Results",
    }.get(phase, phase)

def require_role(role_required: str, role_actual: str):
    if role_required != role_actual:
        st.warning(f"This action is available to **{role_required}** only.")
        st.stop()

def draw_shocks(state: dict, session_code: str):
    if state["shocks_drawn"]:
        return
    if state.get("rng_seed") is not None:
        random.seed(int(state["rng_seed"]))

    supplier_cost_shock = (random.random() < P_SHOCK)
    producer_demand_shock = (random.random() < P_SHOCK)

    c_real = C0 * (C_SHOCK_MULT if supplier_cost_shock else 1.0)
    v_real = V0 * (V_SHOCK_MULT if producer_demand_shock else 1.0)

    state["shocks_drawn"] = True
    state["supplier_cost_shock"] = supplier_cost_shock
    state["producer_demand_shock"] = producer_demand_shock
    state["c_real"] = round(c_real, 2)
    state["v_real"] = round(v_real, 2)

    log_event(session_code, "System", "SHOCKS_REALIZED", {
        "supplier_cost_shock": supplier_cost_shock,
        "producer_demand_shock": producer_demand_shock,
        "c_real": state["c_real"],
        "v_real": state["v_real"],
    })

def settle(state: dict, session_code: str, w: float | None, end_reason: str):
    state["ended"] = True
    state["end_reason"] = end_reason
    state["w_final"] = w

    c = state["c_real"] if state["shocks_drawn"] else C0
    v = state["v_real"] if state["shocks_drawn"] else V0

    supplier = 0.0
    producer = 0.0
    if w is not None:
        supplier = (w - c)
        producer = (v - w)

    supplier -= state["k_paid_each"]
    producer -= state["k_paid_each"]

    state["payoffs"] = {
        "Supplier": round(supplier, 2),
        "Producer": round(producer, 2),
        "c_real": c,
        "v_real": v,
    }
    state["phase"] = "RESULTS"

    log_event(session_code, "System", "SETTLED", {
        "w_final": w,
        "end_reason": end_reason,
        "payoffs": state["payoffs"],
        "k_paid_each": state["k_paid_each"],
    })

# -----------------------------
# App
# -----------------------------
init_db()
st.set_page_config(page_title="TCE Two-Tab Demo", layout="wide")
st.title("TCE Contracting Demo — Two Tabs (Supplier vs Producer)")

# Sidebar: Admin panel + create/join
with st.sidebar:
    st.header("Admin")
    admin_entered = st.text_input("Admin key", type="password")
    is_admin = (admin_entered == ADMIN_KEY and admin_entered != "")

    if is_admin:
        st.success("Admin unlocked")

        st.subheader("Sessions")
        sessions = list_sessions()
        if sessions:
            for s in sessions[:15]:
                st.write(f"**{s['session_code']}** · {iso(s['created_at'])}")
                st.caption(f"Phase: {s['phase']} | Ended: {s['ended']} | Reneg: {s['reneg_triggered']}")
        else:
            st.info("No sessions yet.")

        st.markdown("---")
        st.subheader("Admin actions")
        admin_target = st.text_input("Target session code", value="AB12").upper().strip()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset session"):
                ok = reset_session(admin_target)
                if ok:
                    st.success(f"Reset {admin_target} (state + events wiped).")
                else:
                    st.error("Session not found.")
        with col2:
            if st.button("Delete session"):
                delete_session(admin_target)
                st.success(f"Deleted {admin_target} (session + events).")

        # Export event log
        st.markdown("---")
        st.subheader("Export")
        exp_code = st.text_input("Export session code", value="AB12").upper().strip()
        ev_rows = get_events(exp_code) if exp_code else []

        if exp_code and ev_rows:
            # Build CSV
            lines = ["id,ts_local,actor,event_type,payload_json"]
            for r in ev_rows:
                lines.append(
                    f'{r["id"]},{iso(r["ts"])},{r["actor"]},{r["event_type"]},{json.dumps(json.loads(r["payload_json"]))}'
                )
            csv_data = "\n".join(lines).encode("utf-8")

            st.download_button(
                "Download event log CSV",
                data=csv_data,
                file_name=f"events_{exp_code}.csv",
                mime="text/csv",
            )
        elif exp_code:
            st.info("No events found for that session (or session doesn’t exist).")

        # Export session summary
        sessions = list_sessions()
        if sessions:
            sum_lines = ["session_code,created_at_local,phase,ended,reneg_triggered,w_final,end_reason"]
            for s in sessions:
                sum_lines.append(
                    f'{s["session_code"]},{iso(s["created_at"])},{s["phase"]},{s["ended"]},{s["reneg_triggered"]},{s["w_final"] if s["w_final"] is not None else ""},{(s["end_reason"] or "").replace(",", ";")}'
                )
            summary_csv = "\n".join(sum_lines).encode("utf-8")
            st.download_button(
                "Download sessions summary CSV",
                data=summary_csv,
                file_name="sessions_summary.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.header("Create / Join")

    mode = st.radio("Mode", ["Join existing session", "Create new session"], index=0)

    if mode == "Create new session":
        st.caption("Create a session code and two PINs. Then open two tabs and join with role+PIN.")
        new_code = st.text_input("Session code (e.g., AB12)", value="AB12").upper().strip()
        s_pin = st.text_input("Supplier PIN", value="1111")
        p_pin = st.text_input("Producer PIN", value="2222")
        seed = st.number_input("Optional RNG seed", value=123, step=1)
        use_seed = st.checkbox("Use seed", value=True)

        if st.button("Create session"):
            try:
                create_session(new_code, s_pin, p_pin, int(seed) if use_seed else None)
                st.success(f"Created session **{new_code}**")
            except sqlite3.IntegrityError:
                st.error("That session code already exists. Use a new one or reset/delete it in Admin.")

    st.markdown("---")
    st.subheader("Join existing session")
    session_code = st.text_input("Session code", value="AB12").upper().strip()
    role = st.selectbox("Role", ["Supplier", "Producer"])
    pin = st.text_input("PIN", type="password")

    colA, colB = st.columns(2)
    with colA:
        refresh_clicked = st.button("Refresh")
    with colB:
        auto = st.checkbox("Auto-refresh (2s)", value=False)

    st.markdown("---")
    st.subheader("Public parameters")
    st.write(f"Baseline cost c₀: **{money(C0)}**")
    st.write(f"Baseline value v₀: **{money(V0)}**")
    st.write(f"Shock prob (independent): **{P_SHOCK:.0%}** each")
    st.write(f"Cost shock: **+10%** (Supplier privately observes)")
    st.write(f"Demand shock: **−10%** (Producer privately observes)")
    st.write(f"Renegotiation: either triggers → both pay **{money(K1)}**")

# Lightweight auto-refresh
if auto:
    time.sleep(2)
    st.rerun()

if not session_code:
    st.stop()

sess = load_session(session_code)
if sess is None:
    st.warning("Session not found. Create it in the sidebar (or check the code).")
    st.stop()

# Auth check (demo)
if role == "Supplier":
    if pin != sess["supplier_pin"]:
        st.warning("Enter the correct Supplier PIN to view/act.")
        st.stop()
else:
    if pin != sess["producer_pin"]:
        st.warning("Enter the correct Producer PIN to view/act.")
        st.stop()

state = sess["state"]
phase = state["phase"]

# Top status
c1, c2, c3, c4 = st.columns([1.1, 1.4, 1.4, 1.3])
with c1:
    st.metric("Session", sess["session_code"])
with c2:
    st.metric("Phase", phase_title(phase))
with c3:
    st.metric("Renegotiation triggered?", "Yes" if state["reneg_triggered"] else "No")
with c4:
    st.metric("Reneg cost paid by each", money(state["k_paid_each"]))

# Private panel
st.markdown("---")
st.subheader(f"Private panel — {role}")

if not state["shocks_drawn"]:
    st.info("Shocks not realized yet (they occur after Round 1 agreement).")
else:
    if role == "Supplier":
        if state["supplier_cost_shock"]:
            st.warning(f"Cost shock occurred. Your realized unit cost is **{money(state['c_real'])}**.")
        else:
            st.success(f"No cost shock. Your realized unit cost is **{money(state['c_real'])}**.")
        st.write("You do **not** observe the producer’s demand shock.")
    else:
        if state["producer_demand_shock"]:
            st.warning(f"Demand shock occurred. Your realized unit value is **{money(state['v_real'])}**.")
        else:
            st.success(f"No demand shock. Your realized unit value is **{money(state['v_real'])}**.")
        st.write("You do **not** observe the supplier’s cost shock.")

# Public history panel
st.markdown("---")
st.subheader("Public history")
h1, h2 = st.columns(2)
with h1:
    st.write("**Round 1**")
    st.write(f"Offer w₁: {money(state['w1_offer']) if state['w1_offer'] is not None else '—'}")
    st.write(f"Counter w₁ᶜ: {money(state['w1_counter']) if state['w1_counter'] is not None else '—'}")
    st.write(f"Final w₁ᶠ: {money(state['w1_final']) if state['w1_final'] is not None else '—'}")
    st.write(f"Agreed w₁: {money(state['w1_agreed']) if state['w1_agreed'] is not None else '—'}")
with h2:
    st.write("**Round 2**")
    st.write(f"Offer w₂: {money(state['w2_offer']) if state['w2_offer'] is not None else '—'}")
    st.write(f"Counter w₂ᶜ: {money(state['w2_counter']) if state['w2_counter'] is not None else '—'}")
    st.write(f"Final w₂ᶠ: {money(state['w2_final']) if state['w2_final'] is not None else '—'}")
    st.write(f"Agreed w₂: {money(state['w2_agreed']) if state['w2_agreed'] is not None else '—'}")

# Main actions
st.markdown("---")
st.header("Actions")

if state["ended"]:
    st.success("Game ended.")
    st.write(f"**Reason:** {state['end_reason']}")
    if state["w_final"] is None:
        st.write("**Trade:** No agreement.")
    else:
        st.write(f"**Final price:** {money(state['w_final'])}")

    if state["payoffs"]:
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Supplier payoff", money(state["payoffs"]["Supplier"]))
        with p2:
            st.metric("Producer payoff", money(state["payoffs"]["Producer"]))
        with p3:
            st.metric("Realized surplus (before reneg costs)", money(state["payoffs"]["v_real"] - state["payoffs"]["c_real"]))
    st.stop()

# --- Phase handlers ---
if phase == "R1_SUPPLIER_OFFER":
    require_role("Supplier", role)
    st.write("Supplier proposes input price **w₁**.")
    w = st.number_input("w₁", min_value=W_MIN, max_value=W_MAX, value=10.50, step=0.10)
    if st.button("Submit w₁"):
        state["w1_offer"] = round(float(w), 2)
        state["phase"] = "R1_PRODUCER_RESPONSE"
        save_state(session_code, state)
        log_event(session_code, "Supplier", "R1_OFFER_SUBMITTED", {"w1_offer": state["w1_offer"]})
        st.success("Submitted. Producer tab should refresh.")
        st.stop()

elif phase == "R1_PRODUCER_RESPONSE":
    require_role("Producer", role)
    st.write(f"Supplier proposed **w₁ = {money(state['w1_offer'])}**.")
    choice = st.radio("Response", ["Accept", "Counter", "Reject"], index=1, horizontal=True)
    wc = None
    if choice == "Counter":
        wc = st.number_input("w₁ᶜ", min_value=W_MIN, max_value=W_MAX, value=9.80, step=0.10)

    if st.button("Submit response"):
        if choice == "Accept":
            state["w1_agreed"] = state["w1_offer"]
            state["phase"] = "SHOCKS"
            log_event(session_code, "Producer", "R1_ACCEPTED", {"w1_agreed": state["w1_agreed"]})
        elif choice == "Counter":
            state["w1_counter"] = round(float(wc), 2)
            state["phase"] = "R1_SUPPLIER_FINAL"
            log_event(session_code, "Producer", "R1_COUNTERED", {"w1_counter": state["w1_counter"]})
        else:
            settle(state, session_code, None, "Producer rejected in Round 1 (no agreement).")
            log_event(session_code, "Producer", "R1_REJECTED", {})
        save_state(session_code, state)
        st.success("Submitted. Supplier tab should refresh.")
        st.stop()

elif phase == "R1_SUPPLIER_FINAL":
    require_role("Supplier", role)
    st.write(f"Producer countered with **w₁ᶜ = {money(state['w1_counter'])}**.")
    wf = st.number_input("w₁ᶠ (FINAL)", min_value=W_MIN, max_value=W_MAX,
                         value=clamp(state["w1_counter"] + 0.50, W_MIN, W_MAX), step=0.10)
    if st.button("Submit FINAL offer"):
        state["w1_final"] = round(float(wf), 2)
        state["phase"] = "R1_PRODUCER_FINAL"
        save_state(session_code, state)
        log_event(session_code, "Supplier", "R1_FINAL_SUBMITTED", {"w1_final": state["w1_final"]})
        st.success("Submitted. Producer tab should refresh.")
        st.stop()

elif phase == "R1_PRODUCER_FINAL":
    require_role("Producer", role)
    st.write(f"Supplier FINAL offer is **w₁ᶠ = {money(state['w1_final'])}**.")
    choice = st.radio("Decision", ["Accept", "Reject"], index=0, horizontal=True)
    if st.button("Submit decision"):
        if choice == "Accept":
            state["w1_agreed"] = state["w1_final"]
            state["phase"] = "SHOCKS"
            log_event(session_code, "Producer", "R1_FINAL_ACCEPTED", {"w1_agreed": state["w1_agreed"]})
        else:
            settle(state, session_code, None, "Producer rejected FINAL offer in Round 1 (no agreement).")
            log_event(session_code, "Producer", "R1_FINAL_REJECTED", {})
        save_state(session_code, state)
        st.success("Submitted.")
        st.stop()

elif phase == "SHOCKS":
    st.write(f"Round 1 agreement at **w₁ = {money(state['w1_agreed'])}**.")
    st.info("Click to realize shocks (private) and proceed to renegotiation decision.")
    if st.button("Realize shocks"):
        draw_shocks(state, session_code)
        state["phase"] = "RENEG_DECISION"
        save_state(session_code, state)
        st.success("Shocks realized. Each tab should refresh to see private info.")
        st.stop()

elif phase == "RENEG_DECISION":
    st.warning(f"If either party chooses YES, renegotiation happens and BOTH pay {money(K1)}.")
    if not state["shocks_drawn"]:
        draw_shocks(state, session_code)

    # Each role submits vote separately; once both exist, any YES triggers.
    if role == "Supplier":
        vote = st.radio("Supplier vote", ["No", "Yes"], index=0, horizontal=True)
        if st.button("Submit Supplier vote"):
            state["supplier_wants_reneg"] = vote
            save_state(session_code, state)
            log_event(session_code, "Supplier", "RENEG_VOTE", {"vote": vote})
            st.success("Vote recorded. Producer tab should vote/refresh.")
            st.stop()
    else:
        vote = st.radio("Producer vote", ["No", "Yes"], index=0, horizontal=True)
        if st.button("Submit Producer vote"):
            state["producer_wants_reneg"] = vote
            save_state(session_code, state)
            log_event(session_code, "Producer", "RENEG_VOTE", {"vote": vote})
            st.success("Vote recorded. Supplier tab should refresh.")
            st.stop()

    st.markdown("---")
    st.write("**Votes so far (public):**")
    st.write(f"Supplier: {state['supplier_wants_reneg'] if state['supplier_wants_reneg'] else '—'}")
    st.write(f"Producer: {state['producer_wants_reneg'] if state['producer_wants_reneg'] else '—'}")

    if state["supplier_wants_reneg"] and state["producer_wants_reneg"]:
        wants = (state["supplier_wants_reneg"] == "Yes") or (state["producer_wants_reneg"] == "Yes")
        if wants:
            state["reneg_triggered"] = True
            state["k_paid_each"] += K1
            state["phase"] = "R2_SUPPLIER_OFFER"
            log_event(session_code, "System", "RENEG_TRIGGERED", {"k_paid_each": state["k_paid_each"], "k1": K1})
        else:
            settle(state, session_code, state["w1_agreed"], "No renegotiation; production under Round 1 contract.")
            log_event(session_code, "System", "RENEG_NOT_TRIGGERED", {})
        save_state(session_code, state)
        st.info("Decision resolved. Refresh both tabs.")
        st.stop()

elif phase == "R2_SUPPLIER_OFFER":
    require_role("Supplier", role)
    st.write("Renegotiation triggered. Supplier proposes **w₂**.")
    default = state["w1_agreed"] if state["w1_agreed"] is not None else 10.50
    w = st.number_input("w₂", min_value=W_MIN, max_value=W_MAX, value=float(default), step=0.10)
    if st.button("Submit w₂"):
        state["w2_offer"] = round(float(w), 2)
        state["phase"] = "R2_PRODUCER_RESPONSE"
        save_state(session_code, state)
        log_event(session_code, "Supplier", "R2_OFFER_SUBMITTED", {"w2_offer": state["w2_offer"]})
        st.success("Submitted. Producer tab should refresh.")
        st.stop()

elif phase == "R2_PRODUCER_RESPONSE":
    require_role("Producer", role)
    st.write(f"Supplier proposed **w₂ = {money(state['w2_offer'])}**.")
    choice = st.radio("Response", ["Accept", "Counter", "Reject"], index=1, horizontal=True)
    wc = None
    if choice == "Counter":
        wc = st.number_input("w₂ᶜ", min_value=W_MIN, max_value=W_MAX,
                             value=clamp(state["w2_offer"] - 0.50, W_MIN, W_MAX), step=0.10)

    if st.button("Submit response"):
        if choice == "Accept":
            state["w2_agreed"] = state["w2_offer"]
            log_event(session_code, "Producer", "R2_ACCEPTED", {"w2_agreed": state["w2_agreed"]})
            settle(state, session_code, state["w2_agreed"], "Renegotiated agreement; production under Round 2 contract.")
        elif choice == "Counter":
            state["w2_counter"] = round(float(wc), 2)
            state["phase"] = "R2_SUPPLIER_FINAL"
            log_event(session_code, "Producer", "R2_COUNTERED", {"w2_counter": state["w2_counter"]})
        else:
            log_event(session_code, "Producer", "R2_REJECTED", {})
            settle(state, session_code, None, "Producer rejected in Round 2 (no agreement after renegotiation).")
        save_state(session_code, state)
        st.success("Submitted. Supplier tab should refresh.")
        st.stop()

elif phase == "R2_SUPPLIER_FINAL":
    require_role("Supplier", role)
    st.write(f"Producer countered with **w₂ᶜ = {money(state['w2_counter'])}**.")
    wf = st.number_input("w₂ᶠ (FINAL)", min_value=W_MIN, max_value=W_MAX,
                         value=clamp(state["w2_counter"] + 0.50, W_MIN, W_MAX), step=0.10)
    if st.button("Submit FINAL offer"):
        state["w2_final"] = round(float(wf), 2)
        state["phase"] = "R2_PRODUCER_FINAL"
        save_state(session_code, state)
        log_event(session_code, "Supplier", "R2_FINAL_SUBMITTED", {"w2_final": state["w2_final"]})
        st.success("Submitted. Producer tab should refresh.")
        st.stop()

elif phase == "R2_PRODUCER_FINAL":
    require_role("Producer", role)
    st.write(f"Supplier FINAL offer is **w₂ᶠ = {money(state['w2_final'])}**.")
    choice = st.radio("Decision", ["Accept", "Reject"], index=0, horizontal=True)
    if st.button("Submit decision"):
        if choice == "Accept":
            state["w2_agreed"] = state["w2_final"]
            log_event(session_code, "Producer", "R2_FINAL_ACCEPTED", {"w2_agreed": state["w2_agreed"]})
            settle(state, session_code, state["w2_agreed"], "Renegotiated FINAL accepted; production under Round 2 contract.")
        else:
            log_event(session_code, "Producer", "R2_FINAL_REJECTED", {})
            settle(state, session_code, None, "Producer rejected FINAL in Round 2 (no agreement after renegotiation).")
        save_state(session_code, state)
        st.success("Submitted.")
        st.stop()

elif phase == "RESULTS":
    st.info("Results ready. Refresh.")
else:
    st.error(f"Unknown phase: {phase}")

st.caption("Tip: open two browser tabs to the same URL, join with the same session code, but different roles + PINs.")
