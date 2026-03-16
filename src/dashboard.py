"""
PredictOS — Unified Dashboard
Single page, two tabs: Signals + Progress.
No second file, no navigation, one URL.
"""

import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def bake_dashboard(data: dict, output_path: str = None):
    output_path = output_path or config.DASHBOARD_FILE
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    picks        = data.get("picks", {})
    stats        = data.get("stats", {})
    tweet        = data.get("tweet", "")
    run_date     = data.get("run_date", "")[:10]
    learning     = data.get("learning", {})
    calib        = data.get("calibration", {})
    bankroll     = data.get("bankroll", {})
    paper        = data.get("paper_trading", {})
    picks_json   = json.dumps(picks,  default=str)
    stats_json   = json.dumps(stats,  default=str)

    html = _build_html(picks, stats, tweet, run_date,
                       picks_json, stats_json,
                       learning, calib, bankroll, paper)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _score_bar(score: float, color: str = None) -> str:
    pct = min(100, max(0, score))
    if color is None:
        color = "#00ff88" if pct >= 75 else ("#ffd700" if pct >= 55 else "#ff6b6b")
    return (f'<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div>'
            f'</div>')


def _pillar_dots(pillars: dict) -> str:
    icons = {"momentum":"⚡","volume":"💧","sentiment":"📰","edge":"🎯","decay":"⏳"}
    out = '<div class="pillars">'
    for name, val in pillars.items():
        lvl = "high" if val >= 70 else ("mid" if val >= 40 else "low")
        out += (f'<span class="pillar {lvl}" title="{name.title()}: {val:.0f}">'
                f'{icons.get(name,"•")}<em>{val:.0f}</em></span>')
    return out + '</div>'


def _pick_card(pick: dict, side: str, rank: int, paper_status: str = "accumulating") -> str:
    q          = pick.get("question", "")
    q_short    = q[:72] + "…" if len(q) > 72 else q
    score      = pick.get("composite", 0)
    edge       = pick.get("live_edge", pick.get("edge_pct", 0))
    conf       = pick.get("confidence", "LOW")
    yes_p      = pick.get("yes_price", 0.5)
    model_p    = pick.get("model_prob", 0.5)
    sizing     = pick.get("sizing", {})
    pos_usd    = sizing.get("position_usd", 0)
    kelly_f    = sizing.get("kelly_f", 0)
    days       = pick.get("days_to_res", 0)
    source     = pick.get("source", "?")
    cat        = pick.get("category", "other")
    url        = pick.get("url", "#")
    pillars    = pick.get("pillars", {})
    ml_conf    = pick.get("ml_confidence", 0)
    flags      = pick.get("flags", [])
    ret        = pick.get("return_estimate", {})
    drift      = pick.get("price_drift", 0)
    val_status = pick.get("validation_status", "unavailable")
    cb         = sizing.get("circuit_breaker", "")

    side_color = "#00ff88" if side == "YES" else "#ff4466"
    side_bg    = "rgba(0,255,136,0.06)" if side == "YES" else "rgba(255,68,102,0.06)"
    edge_sign  = f"+{edge:.1f}%" if edge >= 0 else f"{edge:.1f}%"

    # Validation badge
    val_map = {"valid":("✓ LIVE","#00ff88"),"stale":("⚠ STALE","#ffd700"),
               "edge_gone":("✗ EDGE GONE","#ff4466"),"unavailable":("◎ UNVERIFIED","#5a7a9a")}
    val_label, val_color = val_map.get(val_status, ("?","#5a7a9a"))

    # Action label
    if cb:
        act_label, act_color = "⏸ HALTED",    "#5a7a9a"
    elif paper_status == "validated" and conf == "HIGH":
        act_label, act_color = "▶ FOLLOW",     "#00ff88"
    elif paper_status == "validated":
        act_label, act_color = "◎ OPTIONAL",   "#00aaff"
    elif paper_status == "promising" and conf == "HIGH":
        act_label, act_color = "▶ OPTIONAL",   "#ffd700"
    else:
        act_label, act_color = "◎ PAPER",      "#ffd700"

    conf_cls = {"HIGH":"conf-high","MEDIUM":"conf-mid","LOW":"conf-low"}.get(conf,"conf-low")

    # Return estimate block
    ret_html = ""
    if ret:
        net_win = ret.get("net_profit", 0)
        net_loss= ret.get("net_loss", 0)
        ev      = ret.get("expected_value", 0)
        ev_pct  = ret.get("ev_pct", 0)
        stake   = ret.get("stake", pos_usd)
        ci      = ret.get("ci", {})
        ev_col  = "#00ff88" if ev >= 0 else "#ff4466"
        ev_sign = "+" if ev >= 0 else ""
        ret_html = f"""
<div class="ret-box">
  <div class="ret-hdr">RETURN ON ${stake:.0f} STAKE</div>
  <div class="ret-grid">
    <div class="ret-cell win-c"><label>If WIN</label><val>+${net_win:.2f}</val><sub>{ret.get('pct_return',0):.0f}%</sub></div>
    <div class="ret-cell lose-c"><label>If LOSE</label><val>-${abs(net_loss):.2f}</val><sub>full stake</sub></div>
    <div class="ret-cell ev-c"><label>Exp. Value</label><val style="color:{ev_col}">{ev_sign}${ev:.2f}</val><sub>{ev_sign}{ev_pct:.1f}%</sub></div>
  </div>
  <div class="ci-row"><span class="ci-lbl">Range:</span><span class="ci-val">{ci.get('return_low',0):+.0f}% to {ci.get('return_high',0):+.0f}%</span></div>
</div>"""

    flags_html = ""
    if flags:
        flags_html = '<div class="flags">' + "".join(
            f'<span class="flag">{f.replace("_"," ")}</span>' for f in flags[:3]
        ) + '</div>'

    drift_note = f'<div class="drift-note">Price moved {drift:+.1f}¢ since scoring</div>' if abs(drift) > 0.5 else ""

    return f"""
<div class="pick-card" style="border-left:3px solid {side_color};background:{side_bg}">
  <div class="card-header">
    <span class="rank-badge">#{rank}</span>
    <span class="side-badge" style="color:{side_color}">{'▲' if side=='YES' else '▼'} {side}</span>
    <span class="source-tag">{source} · {cat}</span>
    <span class="val-badge" style="color:{val_color};border-color:{val_color}40">{val_label}</span>
    <span class="conf-badge {conf_cls}">{conf}</span>
    <span class="act-badge" style="color:{act_color};border-color:{act_color}40">{act_label}</span>
  </div>
  <a href="{url}" target="_blank" class="question-link">{q_short}</a>
  <div class="score-row">
    <div class="big-score">{score:.0f}<span>/100</span></div>
    {_score_bar(score)}
  </div>
  {_pillar_dots(pillars)}
  <div class="stats-grid">
    <div class="stat"><label>Market</label><value>{yes_p:.1%}</value></div>
    <div class="stat"><label>Model</label><value>{model_p:.1%}</value></div>
    <div class="stat edge-stat"><label>Edge</label><value>{edge_sign}</value></div>
    <div class="stat"><label>Score</label><value>{score:.0f}</value></div>
    <div class="stat stat-hide-mobile"><label>Kelly</label><value>{kelly_f:.1%}</value></div>
    <div class="stat"><label>Suggest</label><value>${pos_usd:.0f}</value></div>
    <div class="stat stat-hide-mobile"><label>Days</label><value>{days:.0f}d</value></div>
  </div>
  {ret_html}
  {drift_note}
  {flags_html}
</div>"""


def _trade_row(t: dict) -> str:
    won  = t.get("won", False)
    pnl  = t.get("actual_pnl", 0) or 0
    exp  = t.get("expected_pnl", 0) or 0
    q    = t.get("question", "")[:48]
    cat  = t.get("category", "")
    dir_ = t.get("direction", "YES")
    stk  = t.get("stake", 0)
    ent  = t.get("entry_price", 0.5)
    date = (t.get("resolved_at") or "")[:10]
    vs   = pnl - exp
    ic   = "✓" if won else "✗"
    ic_c = "#00ff88" if won else "#ff4466"
    pc   = "#00ff88" if pnl >= 0 else "#ff4466"
    vc   = "#00ff88" if vs >= 0 else "#ff4466"
    dc   = "#00ff88" if dir_ == "YES" else "#ff4466"
    return (f"<tr><td style='color:{ic_c}'>{ic}</td><td class='td-q'>{q}</td>"
            f"<td>{cat}</td><td style='color:{dc}'>{dir_}</td>"
            f"<td>${stk:.0f}</td><td>{ent:.0%}</td>"
            f"<td style='color:{pc}'>${pnl:+.2f}</td>"
            f"<td style='color:{vc};font-size:10px'>{vs:+.2f}</td>"
            f"<td style='color:#5a7a9a;font-size:10px'>{date}</td></tr>")


# ─── Main HTML ────────────────────────────────────────────────────────────────

def _build_html(picks, stats, tweet, run_date,
                picks_json, stats_json,
                learning, calib, bankroll, paper) -> str:

    # ── Signals tab data ──────────────────────────────────────────────────────
    yes_picks   = picks.get("yes_picks", [])
    no_picks    = picks.get("no_picks", [])
    macro_plays = picks.get("macro_plays", [])
    universe    = picks.get("universe", 0)
    filtered_in = picks.get("filtered_in", 0)
    rejected    = picks.get("rejected_picks", [])
    corr_flags  = picks.get("correlation_flags", [])
    cb_info     = picks.get("circuit_breaker", {})
    cb_status   = cb_info.get("status", "active")
    cb_reason   = cb_info.get("reason", "")

    total    = stats.get("total", 0)
    wins     = stats.get("wins", 0)
    win_rate = stats.get("win_rate", 0)

    # ── Progress tab data ─────────────────────────────────────────────────────
    paper_stats  = paper.get("stats", {})
    paper_status = paper.get("validation_status", "accumulating")
    paper_broll  = paper.get("bankroll", 1000)
    paper_pnl    = paper.get("total_pnl", 0)
    paper_closed = paper.get("closed_trades", 0)
    paper_open   = paper.get("open_trades", 0)
    n_trades     = paper_stats.get("n", 0)
    p_win_rate   = paper_stats.get("win_rate", 0)
    sharpe       = paper_stats.get("sharpe_ratio", 0)
    total_ret    = paper_stats.get("total_return_pct", 0)
    ev_mae       = paper_stats.get("ev_mae")

    interp       = calib.get("interpretation", {})
    cal_quality  = interp.get("quality", "bootstrapping")
    skill_pct    = interp.get("skill_score_pct", 0)
    brier        = calib.get("brier_score")
    n_res        = calib.get("n_resolved", 0)
    brier_str    = f"{brier:.4f}" if brier else "—"
    cal_note     = interp.get("note", "Accumulating…")

    summary      = learning.get("summary", {})
    recent_wr    = summary.get("recent_10_wr", 0)
    trend        = summary.get("win_trend", "")
    trend_icon   = {"improving":"↑","declining":"↓","stable":"→"}.get(trend, "—")
    trend_col    = {"improving":"#00ff88","declining":"#ff4466","stable":"#ffd700"}.get(trend,"#5a7a9a")

    real_broll   = bankroll.get("current_bankroll", 1000)
    real_drawdown= bankroll.get("drawdown_from_peak", 0)
    real_pnl     = bankroll.get("all_time_pnl", 0)
    real_cb      = bankroll.get("status", "active")

    progress_pct = min(100, int(n_trades / 30 * 100))
    status_map = {
        "accumulating": ("#ffd700", "Accumulating",  "Building history. Paper track only, optional tiny real stakes."),
        "promising":    ("#00aaff", "Promising",      "Looking good. Small real stakes on HIGH picks acceptable."),
        "validated":    ("#00ff88", "Validated ✓",    "Edge proven. Follow picks at suggested size."),
        "inconclusive": ("#ffd700", "Inconclusive",   "Mixed results. Continue paper trading."),
        "failed":       ("#ff4466", "Under Review",   "Underperforming. Paper only until it recovers."),
    }
    st_col, st_label, st_advice = status_map.get(paper_status, status_map["accumulating"])

    # Validation checklist
    def _chk(met, label):
        ic  = "✓" if met else "○"
        col = "#00ff88" if met else "#5a7a9a"
        tc  = "#c8d8e8" if met else "#5a7a9a"
        return f'<div class="crit-row"><span style="color:{col}">{ic}</span><span style="color:{tc}">{label}</span></div>'

    crit_html = (
        _chk(n_trades >= 30,    f"30+ trades ({n_trades}/30)") +
        _chk(p_win_rate >= 0.55,f"Win rate ≥55% ({p_win_rate:.1%})") +
        _chk(total_ret > 0,     f"Positive return ({total_ret:+.1f}%)") +
        _chk(sharpe >= 0.5,     f"Sharpe ≥0.5 ({sharpe:.2f})")
    )

    # Threshold accuracy
    thr_data = learning.get("threshold_accuracy", {})
    thr_html = ""
    for thr, d in thr_data.items():
        wr  = d.get("win_rate", 0)
        n   = d.get("n", 0)
        col = "#00ff88" if wr >= 0.65 else ("#ffd700" if wr >= 0.55 else "#ff4466")
        bar = int(wr * 100)
        thr_html += (f'<div class="thr-row">'
                     f'<span class="thr-lbl">Model ≥{thr.replace(">=","")}</span>'
                     f'<div class="thr-track"><div class="thr-fill" style="width:{bar}%;background:{col}"></div></div>'
                     f'<span class="thr-wr" style="color:{col}">{wr:.0%}</span>'
                     f'<span class="thr-n">n={n}</span></div>')
    if not thr_html:
        thr_html = '<div style="color:#5a7a9a;font-size:11px;padding:6px">Need 10+ resolved picks</div>'

    # Calibration buckets
    cal_html = ""
    for b in calib.get("by_bucket", []):
        pw = int(b["mean_pred"] * 100)
        aw = int(b["actual_freq"] * 100)
        cal_html += (f'<div class="cal-row">'
                     f'<span class="cal-lbl">{b["bucket"]}</span>'
                     f'<div class="cal-track">'
                     f'<div class="cal-pred" style="width:{pw}%"></div>'
                     f'<div class="cal-actual" style="width:{aw}%"></div>'
                     f'</div><span class="cal-n">n={b["n"]}</span></div>')
    if not cal_html:
        cal_html = '<div style="color:#5a7a9a;font-size:11px">Need 5+ resolved trades</div>'

    # Accuracy curve bars
    acc_curve = calib.get("accuracy_curve", [])
    curve_html = ""
    for pt in acc_curve[-15:]:
        wr   = pt["win_rate"]
        h    = int(wr * 100)
        col  = "#00ff88" if wr >= 0.60 else ("#ffd700" if wr >= 0.50 else "#ff4466")
        n_pt = pt["pick_n"]
        curve_html += f'<div class="c-bar" style="height:{h}%;background:{col}" title="{wr:.0%} ({n_pt} picks)"></div>'
    if not curve_html:
        curve_html = '<div style="color:#5a7a9a;font-size:10px;padding:6px">Building…</div>'

    # Daily P&L sparkline
    daily_7    = bankroll.get("daily_pnl_7d", [])
    spark_html = ""
    if daily_7:
        max_abs = max(abs(d.get("pnl",0)) for d in daily_7) or 1
        for d in daily_7:
            pv   = d.get("pnl", 0)
            h    = max(2, int(abs(pv) / max_abs * 36))
            col  = "#00ff88" if pv >= 0 else "#ff4466"
            lbl  = d.get("date","")[-5:]
            spark_html += (f'<div class="sp-col">'
                           f'<div style="height:{h}px;background:{col};width:100%;border-radius:2px 2px 0 0" title="{lbl}: ${pv:+.2f}"></div>'
                           f'<span style="font-size:9px;color:#2a4060">{lbl}</span></div>')

    # Trade history table
    recent_trades = paper.get("recent_trades", [])
    trade_rows = "\n".join(_trade_row(t) for t in reversed(recent_trades)) if recent_trades else \
        '<tr><td colspan="9" style="color:#5a7a9a;text-align:center;padding:20px">No resolved trades yet — check back after picks resolve</td></tr>'

    # Pick cards for signals tab
    paper_status_for_cards = paper_status

    yes_html = "\n".join(_pick_card(p, "YES", i+1, paper_status_for_cards)
                         for i, p in enumerate(yes_picks)) or \
               '<div class="no-picks">No YES picks passed live validation today</div>'

    no_html  = "\n".join(_pick_card(p, "NO",  i+1, paper_status_for_cards)
                         for i, p in enumerate(no_picks)) or \
               '<div class="no-picks">No NO picks today</div>'

    macro_html = "\n".join(_pick_card(p, p.get("direction","YES"), i+1, paper_status_for_cards)
                           for i, p in enumerate(macro_plays)) or \
                 '<div class="no-picks">No macro event plays today</div>'

    # Category win rate table rows
    by_cat = stats.get("by_category", {})
    cat_rows = ""
    for cat, cv in sorted(by_cat.items(), key=lambda x: -x[1].get("win_rate",0)):
        wr  = cv.get("win_rate", 0)
        bw  = int(wr * 100)
        cat_rows += (f'<tr><td class="cat-name">{cat}</td>'
                     f'<td>{cv.get("total",0)}</td><td>{cv.get("wins",0)}</td>'
                     f'<td><div class="mini-bar-track"><div class="mini-bar-fill" style="width:{bw}%"></div></div>'
                     f'<span class="cat-wr">{wr:.1%}</span></td></tr>')

    # Warnings
    warn_html = ""
    if cb_status != "active":
        warn_html += f'<div class="warn-box cb-warn">⚠ CIRCUIT BREAKER {cb_status.upper()}: {cb_reason}</div>'
    for f in corr_flags:
        warn_html += f'<div class="warn-box corr-warn">◈ {f}</div>'
    if rejected:
        warn_html += f'<div class="warn-box info-warn">◎ {len(rejected)} pick(s) rejected by live validation today</div>'

    tweet_escaped = tweet.replace("<","&lt;").replace(">","&gt;")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PredictOS</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#080c10;--bg2:#0d1117;--bg3:#111820;
  --border:#1e2d3d;--b2:#243040;
  --text:#c8d8e8;--dim:#5a7a9a;--mute:#2a4060;
  --yes:#00ff88;--no:#ff4466;--gold:#ffd700;--blue:#00aaff;
  --mono:'Space Mono',monospace;--disp:'Syne',sans-serif;
}}
body{{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;line-height:1.6;min-height:100vh;overflow-x:hidden}}
body::before{{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px);pointer-events:none;z-index:9999}}

/* ── Header ── */
.hdr{{border-bottom:1px solid var(--border);padding:0 32px;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:16px;background:linear-gradient(180deg,#0a1018,var(--bg))}}
.logo{{font-family:var(--disp);font-size:32px;font-weight:800;background:linear-gradient(135deg,var(--yes),var(--blue),#9966ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;padding:22px 0 4px}}
.logo-sub{{font-size:10px;letter-spacing:3px;color:var(--dim);text-transform:uppercase;padding-bottom:22px}}
.hdr-right{{text-align:right;color:var(--dim);font-size:11px;line-height:2;padding:16px 0}}
.live-dot{{display:inline-flex;align-items:center;gap:6px;background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.25);border-radius:100px;padding:5px 14px;font-size:11px;color:var(--yes);letter-spacing:1px}}
.live-dot::before{{content:'';width:6px;height:6px;border-radius:50%;background:var(--yes);animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1;box-shadow:0 0 0 0 rgba(0,255,136,.4)}}50%{{opacity:.6;box-shadow:0 0 0 6px rgba(0,255,136,0)}}}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}

/* ── Tabs ── */
.tab-bar{{display:flex;border-bottom:1px solid var(--border);background:var(--bg2);padding:0 32px}}
.tab{{padding:14px 24px;font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--dim);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;user-select:none}}
.tab:hover{{color:var(--text)}}
.tab.active{{color:var(--text);border-bottom-color:var(--yes)}}
.tab-panel{{display:none}}.tab-panel.active{{display:block}}

/* ── KPI bar ── */
.kpi-bar{{display:grid;grid-template-columns:repeat(5,1fr);border-bottom:1px solid var(--border);background:var(--bg2)}}
.kpi{{padding:18px 22px;border-right:1px solid var(--border);position:relative;overflow:hidden}}
.kpi:last-child{{border-right:none}}
.kpi::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:2px}}
.kpi.g::after{{background:var(--yes)}}.kpi.r::after{{background:var(--no)}}.kpi.au::after{{background:var(--gold)}}.kpi.b::after{{background:var(--blue)}}.kpi.p::after{{background:#9966ff}}
.klbl{{font-size:10px;letter-spacing:2px;color:var(--dim);text-transform:uppercase;display:block}}
.kval{{font-family:var(--disp);font-size:30px;font-weight:800;line-height:1.1;margin:4px 0 2px;display:block}}
.kpi.g .kval{{color:var(--yes)}}.kpi.r .kval{{color:var(--no)}}.kpi.au .kval{{color:var(--gold)}}.kpi.b .kval{{color:var(--blue)}}.kpi.p .kval{{color:#9966ff}}
.ksub{{font-size:10px;color:var(--mute)}}

/* ── Layout ── */
.main{{padding:28px 32px;max-width:1600px;margin:0 auto}}
.sec-hdr{{display:flex;align-items:center;gap:10px;margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid var(--border)}}
.sec-title{{font-family:var(--disp);font-size:17px;font-weight:800;letter-spacing:-.5px}}
.badge{{font-size:9px;letter-spacing:2px;padding:2px 8px;border-radius:3px;text-transform:uppercase}}
.g-badge{{background:rgba(0,255,136,.1);color:var(--yes);border:1px solid rgba(0,255,136,.3)}}
.r-badge{{background:rgba(255,68,102,.1);color:var(--no);border:1px solid rgba(255,68,102,.3)}}
.au-badge{{background:rgba(255,215,0,.1);color:var(--gold);border:1px solid rgba(255,215,0,.3)}}

/* ── Pick cards ── */
.cards-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px;margin-bottom:36px}}
.pick-card{{border-radius:7px;padding:18px;border:1px solid var(--border);animation:fadeUp .4s ease both;transition:transform .2s}}
.pick-card:hover{{transform:translateY(-2px)}}
.card-header{{display:flex;align-items:center;gap:6px;margin-bottom:10px;flex-wrap:wrap}}
.rank-badge{{font-size:11px;color:var(--mute)}}
.side-badge{{font-size:11px;font-weight:700;letter-spacing:1px}}
.source-tag{{font-size:10px;color:var(--mute);text-transform:uppercase;flex:1}}
.val-badge,.act-badge,.conf-badge{{font-size:9px;letter-spacing:.5px;padding:2px 6px;border-radius:3px;border:1px solid;text-transform:uppercase;font-weight:700}}
.conf-high{{background:rgba(0,255,136,.1);color:var(--yes);border-color:rgba(0,255,136,.3)}}
.conf-mid{{background:rgba(255,215,0,.1);color:var(--gold);border-color:rgba(255,215,0,.3)}}
.conf-low{{background:rgba(255,68,102,.1);color:var(--no);border-color:rgba(255,68,102,.3)}}
.question-link{{display:block;color:var(--text);text-decoration:none;font-size:12px;line-height:1.5;margin-bottom:12px;min-height:36px;transition:color .2s}}
.question-link:hover{{color:var(--blue)}}
.score-row{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
.big-score{{font-family:var(--disp);font-weight:800;font-size:26px;color:var(--text);white-space:nowrap}}
.big-score span{{font-size:13px;color:var(--dim);font-weight:400}}
.bar-track{{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:3px;transition:width .8s ease}}
.pillars{{display:flex;gap:5px;margin-bottom:12px;flex-wrap:wrap}}
.pillar{{display:flex;flex-direction:column;align-items:center;gap:2px;font-size:14px;flex:1;min-width:32px;cursor:default}}
.pillar em{{font-style:normal;font-size:10px;font-family:var(--mono)}}
.pillar.high{{opacity:1}}.pillar.low{{opacity:.3}}
.stats-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:4px;overflow:hidden;margin-bottom:8px}}
.stat{{background:var(--bg3);padding:7px 9px;text-align:center}}
.stat label{{display:block;font-size:9px;letter-spacing:1px;color:var(--mute);text-transform:uppercase;margin-bottom:1px}}
.stat value{{display:block;font-size:12px;font-weight:700;color:var(--text)}}
.edge-stat value{{color:var(--gold)}}
.no-picks{{color:var(--mute);font-size:12px;padding:20px;text-align:center;border:1px dashed var(--border);border-radius:4px;margin-bottom:36px}}

/* ── Return box ── */
.ret-box{{background:var(--bg3);border:1px solid var(--b2);border-radius:5px;padding:10px;margin:8px 0}}
.ret-hdr{{font-size:9px;letter-spacing:2px;color:var(--mute);text-transform:uppercase;margin-bottom:8px}}
.ret-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);border-radius:3px;overflow:hidden;margin-bottom:6px}}
.ret-cell{{background:var(--bg2);padding:7px;text-align:center}}
.win-c{{border-top:2px solid var(--yes)}}.lose-c{{border-top:2px solid var(--no)}}.ev-c{{border-top:2px solid var(--gold)}}
.ret-cell label{{display:block;font-size:9px;color:var(--mute);text-transform:uppercase;margin-bottom:2px}}
.ret-cell val{{display:block;font-family:var(--disp);font-size:15px;font-weight:800}}
.win-c val{{color:var(--yes)}}.lose-c val{{color:var(--no)}}
.ret-cell sub{{display:block;font-size:9px;color:var(--mute);margin-top:1px}}
.ci-row{{font-size:10px;color:var(--dim)}}
.ci-lbl{{color:var(--mute)}}.ci-val{{color:var(--blue);margin-left:6px;font-weight:700}}
.drift-note{{font-size:10px;color:var(--gold);margin-top:4px}}
.flags{{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px}}
.flag{{font-size:9px;padding:2px 6px;background:rgba(255,215,0,.05);border:1px solid rgba(255,215,0,.2);color:var(--gold);border-radius:2px}}

/* ── Warnings ── */
.warn-box{{border-radius:4px;padding:10px 14px;margin-bottom:10px;font-size:11px}}
.cb-warn{{background:rgba(255,68,102,.06);border:1px solid rgba(255,68,102,.25);color:var(--no)}}
.corr-warn{{background:rgba(255,215,0,.06);border:1px solid rgba(255,215,0,.2);color:var(--gold)}}
.info-warn{{background:rgba(0,170,255,.06);border:1px solid rgba(0,170,255,.2);color:var(--blue)}}

/* ── Bottom panels ── */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}}
.three-col{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:28px}}
.panel{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:20px}}
.p-title{{font-size:10px;letter-spacing:2px;color:var(--dim);text-transform:uppercase;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid var(--border)}}
.win-table{{width:100%;border-collapse:collapse}}
.win-table th{{font-size:9px;letter-spacing:1px;color:var(--mute);text-transform:uppercase;text-align:left;padding:0 8px 8px;border-bottom:1px solid var(--border)}}
.win-table td{{padding:7px 8px;font-size:12px;border-bottom:1px solid var(--border)}}
.win-table tr:last-child td{{border-bottom:none}}
.cat-name{{color:var(--blue);text-transform:capitalize}}
.mini-bar-track{{display:inline-block;width:52px;height:4px;background:var(--border);border-radius:2px;vertical-align:middle;margin-right:6px;overflow:hidden}}
.mini-bar-fill{{height:100%;background:var(--yes);border-radius:2px}}
.cat-wr{{color:var(--yes)}}
.tweet-box{{background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:14px;font-size:12px;line-height:1.7;white-space:pre-wrap;word-break:break-word}}
.copy-btn{{margin-top:10px;background:rgba(0,170,255,.1);border:1px solid rgba(0,170,255,.3);color:var(--blue);padding:5px 14px;font-family:var(--mono);font-size:11px;letter-spacing:1px;cursor:pointer;border-radius:3px;transition:background .2s}}
.copy-btn:hover{{background:rgba(0,170,255,.2)}}

/* ── Progress tab ── */
.status-banner{{margin-bottom:24px;padding:20px 24px;border-radius:8px;border:1px solid;display:grid;grid-template-columns:1fr auto;align-items:center;gap:20px}}
.st-label{{font-family:var(--disp);font-size:20px;font-weight:800}}
.st-advice{{font-size:11px;margin-top:3px;opacity:.8}}
.prog-track{{height:5px;background:var(--border);border-radius:3px;overflow:hidden;width:260px;margin-top:10px}}
.prog-fill{{height:100%;border-radius:3px;transition:width .8s}}
.prog-lbl{{font-size:10px;color:var(--dim);margin-top:4px;letter-spacing:1px}}
.paper-pnl{{text-align:right}}
.paper-pnl .big{{font-family:var(--disp);font-size:34px;font-weight:800}}
.paper-pnl .sub{{font-size:10px;color:var(--dim);margin-top:2px}}
.crit-row{{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border);font-size:11px}}
.crit-row:last-child{{border-bottom:none}}
.cal-row{{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:11px}}
.cal-lbl{{width:72px;color:var(--dim);flex-shrink:0}}
.cal-track{{flex:1;height:8px;background:var(--border);border-radius:4px;overflow:hidden;position:relative}}
.cal-pred{{position:absolute;top:0;height:100%;background:var(--blue);opacity:.4;border-radius:4px}}
.cal-actual{{position:absolute;top:0;height:100%;background:var(--yes);border-radius:4px}}
.cal-n{{width:36px;text-align:right;color:var(--mute);font-size:10px}}
.curve-wrap{{display:flex;align-items:flex-end;gap:3px;height:56px;margin-top:8px}}
.c-bar{{flex:1;border-radius:2px 2px 0 0;min-height:3px}}
.thr-row{{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:11px}}
.thr-lbl{{width:96px;color:var(--dim);flex-shrink:0}}
.thr-track{{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}}
.thr-fill{{height:100%;border-radius:2px;transition:width .8s}}
.thr-wr{{width:32px;text-align:right;font-weight:700}}
.thr-n{{width:32px;text-align:right;color:var(--mute);font-size:10px}}
.sp-row{{display:flex;align-items:flex-end;gap:5px;height:44px;margin:8px 0}}
.sp-col{{display:flex;flex-direction:column;align-items:center;gap:3px;flex:1}}
.trade-table{{width:100%;border-collapse:collapse;font-size:11px}}
.trade-table th{{font-size:9px;letter-spacing:1px;color:var(--mute);text-transform:uppercase;text-align:left;padding:0 7px 7px;border-bottom:1px solid var(--border)}}
.trade-table td{{padding:6px 7px;border-bottom:1px solid var(--border)}}
.trade-table tr:last-child td{{border-bottom:none}}
.td-q{{max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.honest-note{{background:rgba(255,215,0,.04);border:1px solid rgba(255,215,0,.12);border-left:3px solid var(--gold);border-radius:4px;padding:12px;font-size:11px;color:var(--dim);line-height:1.7;margin-top:12px}}

/* ── Footer ── */
.footer{{border-top:1px solid var(--border);padding:16px 32px;display:flex;justify-content:space-between;color:var(--mute);font-size:11px}}

/* Hide on mobile */
.stat-hide-mobile{{}}

/* ── Responsive ── */

/* Tablet: 600–900px */
@media(max-width:900px){{
  /* Header */
  .hdr{{grid-template-columns:1fr auto;padding:14px 16px;gap:8px}}
  .hdr-right{{display:none}}

  /* KPI bar: 3 across */
  .kpi-bar{{grid-template-columns:repeat(3,1fr)}}
  .kpi:nth-child(3){{border-right:none}}
  .kpi:nth-child(4){{border-right:1px solid var(--border)}}
  .kpi:nth-child(n+4){{border-top:1px solid var(--border)}}
  .kval{{font-size:24px}}

  /* Tabs */
  .tab-bar{{padding:0 16px}}
  .tab{{padding:12px 16px;font-size:10px}}

  /* Layout */
  .main{{padding:16px}}
  .two-col,.three-col{{grid-template-columns:1fr}}

  /* Cards */
  .cards-grid{{grid-template-columns:1fr}}

  /* Stats grid: 4 across stays but smaller */
  .stats-grid{{grid-template-columns:repeat(4,1fr)}}
  .stat label{{font-size:8px}}
  .stat value{{font-size:11px}}

  /* Progress KPI inside tab */
  .kpi-bar[style*="border-radius"]{{grid-template-columns:repeat(3,1fr)}}
}}

/* Mobile: <600px — everything single column */
@media(max-width:600px){{
  /* Header: stack vertically */
  .hdr{{grid-template-columns:1fr;padding:12px 14px}}
  .logo{{font-size:24px;padding:14px 0 2px}}
  .logo-sub{{font-size:9px;padding-bottom:12px}}
  .live-dot{{font-size:10px;padding:4px 10px}}

  /* KPI bar: 2x2 + 1 */
  .kpi-bar{{grid-template-columns:repeat(2,1fr)}}
  .kpi{{padding:12px 14px}}
  .kpi:nth-child(2){{border-right:none}}
  .kpi:nth-child(3){{border-right:1px solid var(--border);border-top:1px solid var(--border)}}
  .kpi:nth-child(4){{border-right:none;border-top:1px solid var(--border)}}
  .kpi:nth-child(5){{grid-column:1/-1;border-right:none;border-top:1px solid var(--border)}}
  .kval{{font-size:22px}}
  .klbl{{font-size:9px}}
  .ksub{{font-size:9px}}

  /* Tabs: full width, equal split */
  .tab-bar{{padding:0;display:grid;grid-template-columns:1fr 1fr}}
  .tab{{text-align:center;padding:12px 8px;font-size:10px}}

  /* Main padding */
  .main{{padding:12px}}

  /* Pick cards: full width, smaller padding */
  .pick-card{{padding:14px}}
  .big-score{{font-size:22px}}

  /* Card header: wrap tightly */
  .card-header{{gap:4px}}
  .val-badge,.act-badge,.conf-badge{{font-size:8px;padding:1px 5px}}
  .source-tag{{width:100%;order:10;font-size:9px}}

  /* Stats grid: 4 cols compressed */
  .stats-grid{{grid-template-columns:repeat(4,1fr)}}
  .stat-hide-mobile{{display:none}}
  .stat{{padding:5px 4px}}
  .stat label{{font-size:8px}}
  .stat value{{font-size:10px}}

  /* Return box */
  .ret-box{{padding:8px}}
  .ret-hdr{{font-size:8px}}
  .ret-grid{{grid-template-columns:repeat(3,1fr)}}
  .ret-cell{{padding:5px 4px}}
  .ret-cell label{{font-size:8px}}
  .ret-cell val{{font-size:13px}}
  .ret-cell sub{{font-size:8px}}

  /* Pillars: smaller */
  .pillar{{font-size:13px;min-width:28px}}
  .pillar em{{font-size:9px}}

  /* Bottom panels */
  .two-col,.three-col{{grid-template-columns:1fr;gap:12px}}
  .panel{{padding:14px}}

  /* Win table: compact */
  .win-table td,.win-table th{{padding:5px 6px;font-size:10px}}
  .mini-bar-track{{width:36px}}

  /* Progress tab */
  .status-banner{{grid-template-columns:1fr;gap:12px;padding:14px}}
  .paper-pnl{{text-align:left}}
  .paper-pnl .big{{font-size:28px}}
  .prog-track{{width:100%}}

  /* Calibration rows */
  .cal-row{{gap:6px}}
  .cal-lbl{{width:60px;font-size:10px}}

  /* Threshold rows */
  .thr-lbl{{width:80px;font-size:10px}}

  /* Trade table: hide less important cols */
  .trade-table th:nth-child(3),
  .trade-table td:nth-child(3),
  .trade-table th:nth-child(8),
  .trade-table td:nth-child(8){{display:none}}
  .td-q{{max-width:120px}}

  /* Honest note */
  .honest-note{{font-size:10px}}

  /* Section headers */
  .sec-title{{font-size:15px}}
  .badge{{font-size:8px}}

  /* Warning boxes */
  .warn-box{{font-size:10px;padding:8px 10px}}

  /* Tweet box */
  .tweet-box{{font-size:11px}}

  /* Footer */
  .footer{{flex-direction:column;gap:4px;padding:12px 14px;font-size:10px}}
}}

/* Very small: <380px (iPhone SE etc) */
@media(max-width:380px){{
  .hdr{{padding:10px 12px}}
  .logo{{font-size:20px}}
  .kpi-bar{{grid-template-columns:1fr 1fr}}
  .kpi{{padding:10px 12px}}
  .kval{{font-size:20px}}
  .main{{padding:10px}}
  .pick-card{{padding:12px}}
  .ret-grid{{grid-template-columns:1fr}}
  .ret-cell{{border-top:none;border-left:2px solid}}
  .win-c{{border-left-color:var(--yes)}}
  .lose-c{{border-left-color:var(--no)}}
  .ev-c{{border-left-color:var(--gold)}}
  .stats-grid{{grid-template-columns:repeat(3,1fr)}}
}}
</style>
</head>
<body>

<header class="hdr">
  <div>
    <div class="logo">PredictOS</div>
    <div class="logo-sub">Prediction Market Signal System</div>
  </div>
  <div><div class="live-dot">LIVE SIGNALS</div></div>
  <div class="hdr-right">
    <div style="color:var(--blue)">{run_date}</div>
    <div>{universe} markets scanned · {filtered_in} passed</div>
    <div>Polymarket · Kalshi · Manifold</div>
  </div>
</header>

<!-- KPI bar -->
<div class="kpi-bar">
  <div class="kpi g">
    <span class="klbl">YES picks</span>
    <span class="kval">{len(yes_picks)}</span>
    <span class="ksub">High-conviction longs</span>
  </div>
  <div class="kpi r">
    <span class="klbl">NO picks</span>
    <span class="kval">{len(no_picks)}</span>
    <span class="ksub">Fading overpriced</span>
  </div>
  <div class="kpi au">
    <span class="klbl">Win rate</span>
    <span class="kval">{win_rate:.1%}</span>
    <span class="ksub">{total} resolved picks</span>
  </div>
  <div class="kpi b">
    <span class="klbl">Paper trades</span>
    <span class="kval">{paper_closed}</span>
    <span class="ksub">{paper_open} open · {paper_status}</span>
  </div>
  <div class="kpi p">
    <span class="klbl">Paper P&L</span>
    <span class="kval">{'+'if paper_pnl>=0 else ''}{paper_pnl:.0f}</span>
    <span class="ksub">from $1,000 virtual</span>
  </div>
</div>

<!-- Tabs -->
<div class="tab-bar">
  <div class="tab active" onclick="switchTab('signals',this)">Signals</div>
  <div class="tab" onclick="switchTab('progress',this)">Progress</div>
</div>

<!-- ═══════════════ SIGNALS TAB ═══════════════ -->
<div id="tab-signals" class="tab-panel active">
<div class="main">

  {warn_html}

  <div class="sec-hdr">
    <div class="sec-title">▲ YES Picks</div>
    <div class="badge g-badge">Top {len(yes_picks)}</div>
  </div>
  <div class="cards-grid">{yes_html}</div>

  <div class="sec-hdr">
    <div class="sec-title">▼ NO Picks</div>
    <div class="badge r-badge">Fading overpriced</div>
  </div>
  <div class="cards-grid">{no_html}</div>

  <div class="sec-hdr">
    <div class="sec-title">◈ Macro Event Plays</div>
    <div class="badge au-badge">FX equivalent</div>
  </div>
  <div class="cards-grid">{macro_html}</div>

  <div class="two-col">
    <div class="panel">
      <div class="p-title">Win rate by category</div>
      <table class="win-table">
        <thead><tr><th>Category</th><th>N</th><th>Wins</th><th>Win rate</th></tr></thead>
        <tbody>{cat_rows or '<tr><td colspan="4" style="color:var(--mute);text-align:center;padding:16px">No resolved picks yet</td></tr>'}</tbody>
      </table>
    </div>
    <div class="panel">
      <div class="p-title">Daily signal tweet</div>
      <div class="tweet-box" id="tweet-txt">{tweet_escaped}</div>
      <button class="copy-btn" onclick="copyTweet()">COPY</button>
    </div>
  </div>

</div>
</div>

<!-- ═══════════════ PROGRESS TAB ═══════════════ -->
<div id="tab-progress" class="tab-panel">
<div class="main">

  <!-- Status banner -->
  <div class="status-banner" style="border-color:{st_col}40;background:{st_col}08">
    <div>
      <div class="st-label" style="color:{st_col}">{st_label}</div>
      <div class="st-advice">{st_advice}</div>
      <div class="prog-track"><div class="prog-fill" style="width:{progress_pct}%;background:{st_col}"></div></div>
      <div class="prog-lbl">{n_trades}/30 trades · {progress_pct}% to validation</div>
    </div>
    <div class="paper-pnl">
      <div style="font-size:10px;color:var(--dim);letter-spacing:1px;text-transform:uppercase">Paper P&L</div>
      <div class="big" style="color:{'var(--yes)'if paper_pnl>=0 else 'var(--no)'}">{'+'if paper_pnl>=0 else ''}{paper_pnl:.2f}</div>
      <div class="sub">from $1,000 virtual bankroll</div>
    </div>
  </div>

  <!-- Top KPIs -->
  <div class="kpi-bar" style="border-radius:8px;border:1px solid var(--border);margin-bottom:24px">
    <div class="kpi {'g'if p_win_rate>=0.55 else 'au'if p_win_rate>=0.45 else 'r'}">
      <span class="klbl">Paper win rate</span>
      <span class="kval">{p_win_rate:.1%}</span>
      <span class="ksub">need ≥55% · trend {trend_icon}</span>
    </div>
    <div class="kpi {'g'if total_ret>0 else 'r'}">
      <span class="klbl">Paper return</span>
      <span class="kval">{'+'if total_ret>=0 else ''}{total_ret:.1f}%</span>
      <span class="ksub">Sharpe {sharpe:.2f} · need ≥0.5</span>
    </div>
    <div class="kpi b">
      <span class="klbl">Model skill</span>
      <span class="kval">{skill_pct:.0f}%</span>
      <span class="ksub">Brier {brier_str} · {cal_quality}</span>
    </div>
    <div class="kpi {'g'if real_cb=='active' else 'au'if real_cb=='reduced' else 'r'}">
      <span class="klbl">Circuit breaker</span>
      <span class="kval" style="font-size:16px;margin-top:6px">{real_cb.upper()}</span>
      <span class="ksub">Drawdown {real_drawdown:.1f}%</span>
    </div>
    <div class="kpi au">
      <span class="klbl">EV accuracy</span>
      <span class="kval">{'$'+f'{ev_mae:.2f}'if ev_mae else '—'}</span>
      <span class="ksub">avg P&L prediction error</span>
    </div>
  </div>

  <div class="three-col">

    <!-- Validation checklist -->
    <div class="panel">
      <div class="p-title">Validation checklist</div>
      <div style="background:var(--bg3);border:1px solid var(--b2);border-radius:5px;padding:12px;margin-bottom:12px">
        {crit_html}
      </div>
      <div style="font-size:10px;color:var(--dim);line-height:1.7">
        All 4 must be green before real-money sizing.
        Until then: paper only, or optional small stakes on HIGH confidence picks only.
      </div>
      <div class="honest-note">
        ⚡ Realistic target is 65–70% win rate — not 98%.
        A calibrated 65% with Kelly sizing compounds into real profits over time.
      </div>
    </div>

    <!-- Win rate by confidence + calibration -->
    <div class="panel">
      <div class="p-title">Win rate by model confidence</div>
      {thr_html}
      <div style="margin-top:16px">
        <div class="p-title" style="margin-top:0">Probability calibration</div>
        <div style="font-size:10px;color:var(--dim);margin-bottom:8px">Blue = predicted · Green = actual resolution rate</div>
        {cal_html}
        <div style="font-size:10px;color:var(--dim);margin-top:6px">{cal_note}</div>
      </div>
    </div>

    <!-- Accuracy over time + real bankroll -->
    <div class="panel">
      <div class="p-title">Win rate over time (rolling 10 picks)</div>
      <div class="curve-wrap">{curve_html}</div>
      <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--mute);margin-bottom:16px">
        <span>older</span><span>recent →</span>
      </div>
      <div class="p-title">Real bankroll — daily P&L</div>
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">
        <span style="color:var(--dim)">Balance</span>
        <span style="color:var(--yes);font-weight:700">${real_broll:.2f}</span>
      </div>
      <div class="sp-row">{spark_html or '<div style="color:var(--mute);font-size:10px">No trades yet</div>'}</div>
      <div style="font-size:10px;color:var(--dim);margin-top:4px">All-time P&L: {'+'if real_pnl>=0 else ''}{real_pnl:.2f}</div>
    </div>

  </div>

  <!-- Trade history -->
  <div class="panel">
    <div class="p-title">Paper trade history</div>
    <div style="overflow-x:auto">
      <table class="trade-table">
        <thead><tr><th></th><th>Market</th><th>Cat</th><th>Dir</th><th>Stake</th><th>Entry</th><th>Actual P&L</th><th>vs Predicted</th><th>Date</th></tr></thead>
        <tbody>{trade_rows}</tbody>
      </table>
    </div>
  </div>

</div>
</div>

<footer class="footer">
  <div>PredictOS — signals only, not financial advice. Paper trading ≠ guaranteed live returns.</div>
  <div>Updated {run_date}</div>
</footer>

<script>
window.PREDICTOS = {{picks:{picks_json},stats:{stats_json}}};

function switchTab(name, el) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  el.classList.add('active');
}}

function copyTweet() {{
  navigator.clipboard.writeText(document.getElementById('tweet-txt').innerText).then(() => {{
    const b = document.querySelector('.copy-btn');
    b.textContent = 'COPIED ✓';
    setTimeout(() => b.textContent = 'COPY', 2000);
  }});
}}

document.addEventListener('DOMContentLoaded', () => {{
  document.querySelectorAll('.bar-fill,.mini-bar-fill,.prog-fill,.thr-fill').forEach(el => {{
    const w = el.style.width; el.style.width = '0';
    setTimeout(() => el.style.width = w, 80);
  }});
}});
</script>
</body>
</html>"""
