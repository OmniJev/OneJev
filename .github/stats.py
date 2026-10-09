"""Count OneJev downloads on Hugging Face and draw them like star-history.com.

    python .github/stats.py

Appends today's numbers to assets/downloads.csv and redraws assets/downloads.svg.
"""
import base64
import csv
import datetime
import io
import json
import math
import os
import re
import urllib.request

from fontTools.ttLib import TTFont

API = "https://huggingface.co/api/models"
OWNER = "OmniJev"
FAMILY = "OmniJev/OneJev"
KINDS = ("quantized", "finetune", "adapter", "merge")
LEDGER = "assets/downloads.csv"
FONT = "https://cdn.jsdelivr.net/npm/chart.xkcd@2.0.12/dist/chart.xkcd.min.js"
AVATAR = "https://github.com/OmniJev.png?size=44"
TITLE = "Download History"
LEGEND = "omnijev/onejev"
Y_LABEL = "Hugging Face Downloads"
W, H = 800, 533.333
LEFT, TOP, RIGHT, BOTTOM = 70, 60, 30, 50
COLOURS = {"bg": "#fff", "ink": "#000", "line": "#d45bb6"}
STEPS = (1, 2, 7, 30, 91, 365)


def fetch(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "onejev-stats", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), r.headers.get_content_type()


def get(url):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return json.loads(fetch(url, {"Authorization": f"Bearer {token}"} if token else None)[0])


def family():
    fields = "expand[]=downloadsAllTime&expand[]=createdAt"
    rows = {}
    for m in get(f"{API}?author={OWNER}&{fields}"):
        if m["id"].startswith(FAMILY):
            rows[m["id"]] = ("ours", m)
    for rid in list(rows):
        for kind in KINDS:
            for m in get(f"{API}?filter=base_model:{kind}:{rid}&{fields}"):
                if m["id"] not in rows:
                    rows[m["id"]] = ("community", m)
    return rows


def update(rows):
    old = []
    if os.path.exists(LEDGER):
        with open(LEDGER, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    known = {r["repo"] for r in old}
    out = [r for r in old if r["date"] != today]
    for rid, (side, m) in sorted(rows.items()):
        if rid not in known:
            born = (m.get("createdAt") or "")[:10]
            if born and born < today:
                out.append({"date": born, "repo": rid, "side": side, "downloads": "0"})
        out.append({"date": today, "repo": rid, "side": side,
                    "downloads": str(m.get("downloadsAllTime") or 0)})
    out.sort(key=lambda r: (r["date"], r["repo"]))
    with open(LEDGER, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "repo", "side", "downloads"])
        w.writeheader()
        w.writerows(out)
    return out


def series(ledger):
    per_repo = {}
    for r in ledger:
        per_repo.setdefault(r["repo"], {})[r["date"]] = int(r["downloads"])
    points = []
    for d in sorted({r["date"] for r in ledger}):
        total = 0
        for counts in per_repo.values():
            seen = [v for day, v in sorted(counts.items()) if day <= d]
            total += seen[-1] if seen else 0
        if not points or total != points[-1][1]:
            points.append((datetime.date.fromisoformat(d), total))
    return points


def x_ticks(first, last):
    target = max((last - first).days, 1) / 5
    step = STEPS[-1]
    for lo, hi in zip(STEPS, STEPS[1:]):
        if target < hi:
            step = lo if target / lo < hi / target else hi
            break
    keep = {1: lambda d: True,
            2: lambda d: (d.day - 1) % 2 == 0,
            7: lambda d: d.weekday() == 6,
            30: lambda d: d.day == 1,
            91: lambda d: d.day == 1 and d.month % 3 == 1,
            365: lambda d: d.day == 1 and d.month == 1}[step]
    days = [first + datetime.timedelta(days=i) for i in range((last - first).days + 1)]
    return [d for d in days if keep(d)]


def x_label(d):
    if d.day == 1:
        return d.strftime("%Y") if d.month == 1 else d.strftime("%B")
    return d.strftime("%b %d") if d.weekday() == 6 else d.strftime("%a %d")


def y_ticks(top):
    raw = top / 10
    power = 10 ** math.floor(math.log10(raw))
    error = raw / power
    factor = 10 if error >= 50 ** .5 else 5 if error >= 10 ** .5 else 2 if error >= 2 ** .5 else 1
    step = max(int(factor * power), 1)
    return list(range(0, int(top) + 1, step))


def y_label(v):
    if v == 0:
        return " "
    return f"{v / 1000:g}k" if v >= 1000 else str(v)


def monotone(pts):
    if len(pts) < 3:
        return "M" + "L".join(f"{x:.3f},{y:.3f}" for x, y in pts)
    sign = lambda v: -1 if v < 0 else 1
    s = [(y1 - y0) / (x1 - x0) for (x0, y0), (x1, y1) in zip(pts, pts[1:])]
    m = [0.0] * len(pts)
    for i in range(1, len(pts) - 1):
        h0, h1 = pts[i][0] - pts[i - 1][0], pts[i + 1][0] - pts[i][0]
        p = (s[i - 1] * h1 + s[i] * h0) / (h0 + h1)
        m[i] = (sign(s[i - 1]) + sign(s[i])) * min(abs(s[i - 1]), abs(s[i]), 0.5 * abs(p))
    m[0] = (3 * s[0] - m[1]) / 2
    m[-1] = (3 * s[-1] - m[-2]) / 2
    out = [f"M{pts[0][0]:.3f},{pts[0][1]:.3f}"]
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(pts, pts[1:])):
        dx = (x1 - x0) / 3
        out.append(f"C{x0 + dx:.3f},{y0 + dx * m[i]:.3f},{x1 - dx:.3f},{y1 - dx * m[i + 1]:.3f},{x1:.3f},{y1:.3f}")
    return "".join(out)


def chart(points, c, font, measure, avatar):
    w, h = W - LEFT - RIGHT, H - TOP - BOTTOM
    first, last = points[0][0], points[-1][0]
    top = max(v for _, v in points)
    px = lambda d: (d - first).days / max((last - first).days, 1) * w
    py = lambda v: h - v / top * h
    tick = f"font-family:xkcd;font-size:16px;fill:{c['ink']}"
    xs = "".join(f'<text y="6" fill="currentColor" class="tick" dy=".71em" style="{tick}" '
                 f'transform="translate({px(d):.3f} {h})">{x_label(d)}</text>' for d in x_ticks(first, last))
    ys = "".join(f'<g class="tick"><path stroke="currentColor" d="M0 {py(v) + .5:.3f}h-1"/>'
                 f'<text x="-7" fill="currentColor" dy=".32em" style="{tick}" '
                 f'transform="translate(0 {py(v) + .5:.3f})">{y_label(v)}</text></g>' for v in y_ticks(top))
    line = monotone([(px(d), py(v)) for d, v in points])
    box = measure(LEGEND, 15) + 28
    head = W / 2 - measure(TITLE, 20) / 2 - 30
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'style="stroke-width:3;font-family:xkcd;background:{c["bg"]}">'
        f'<defs><style>@font-face{{font-family:"xkcd";src:url(data:application/font-woff;charset=utf-8;base64,{font})'
        f'format("woff")}}</style><clipPath id="clip-circle-title"><circle cx="{head + 11:.3f}" cy="23" r="11"/>'
        f'</clipPath></defs>'
        f'<filter id="xkcdify" width="100%" height="100%" x="-5" y="-5" filterUnits="userSpaceOnUse">'
        f'<feTurbulence baseFrequency=".05" result="noise" type="fractalNoise"/>'
        f'<feDisplacementMap in="SourceGraphic" in2="noise" scale="5" xChannelSelector="R" yChannelSelector="G"/></filter>'
        f'<g pointer-events="all" transform="translate({LEFT} {TOP})">'
        f'<g fill="none" class="xaxis" font-family="sans-serif" font-size="10" text-anchor="middle">'
        f'<path stroke="currentColor" d="M.5.5h{w}" class="domain" filter="url(#xkcdify)" style="stroke:{c["ink"]}" '
        f'transform="translate(0 {h})"/>{xs}</g>'
        f'<g fill="none" class="yaxis" font-family="sans-serif" font-size="10" text-anchor="end">'
        f'<path stroke="currentColor" d="M-1 {h + .5:.3f}H.5V.5H-1" class="domain" filter="url(#xkcdify)" '
        f'style="stroke:{c["ink"]}"/>{ys}</g>'
        f'<path fill="none" stroke="{c["line"]}" d="{line}" class="xkcd-chart-xyline" filter="url(#xkcdify)"/>'
        f'<rect width="{box:.3f}" height="32" fill-opacity=".85" stroke="{c["ink"]}" stroke-width="2" '
        f'filter="url(#xkcdify)" rx="5" ry="5" style="fill:{c["bg"]}" transform="translate(8 5)"/>'
        f'<g transform="translate(8 5)"><rect width="8" height="8" x="7" y="12" filter="url(#xkcdify)" rx="2" ry="2" '
        f'style="fill:{c["line"]}"/><text x="21" y="20" style="font-size:15px;fill:{c["ink"]}">{LEGEND}</text></g></g>'
        f'<text x="50%" y="30" style="font-size:20px;font-weight:700;fill:{c["ink"]}" text-anchor="middle">{TITLE}</text>'
        f'<image width="22" height="22" x="{head:.3f}" y="12" clip-path="url(#clip-circle-title)" href="{avatar}"/>'
        f'<text x="50%" y="{H - 10:.3f}" style="font-size:17px;fill:{c["ink"]}" text-anchor="middle">Date</text>'
        f'<text x="{-(TOP + h / 2):.3f}" y="20" dy=".75em" style="font-size:17px;fill:{c["ink"]}" '
        f'text-anchor="middle" transform="rotate(-90)">{Y_LABEL}</text></svg>\n')


def main():
    ledger = update(family())
    points = series(ledger)
    font = re.search(r"base64,([A-Za-z0-9+/=]{1000,})", fetch(FONT)[0].decode()).group(1)
    tt = TTFont(io.BytesIO(base64.b64decode(font)))
    cmap, hmtx, em = tt.getBestCmap(), tt["hmtx"], tt["head"].unitsPerEm
    measure = lambda text, size: sum(hmtx[cmap[ord(ch)]][0] for ch in text) / em * size
    body, kind = fetch(AVATAR)
    avatar = f"data:{kind};base64,{base64.b64encode(body).decode()}"
    with open("assets/downloads.svg", "w", encoding="utf-8") as f:
        f.write(chart(points, COLOURS, font, measure, avatar))
    print(f"{points[-1][0]}: {points[-1][1]:,} downloads over {len({r['repo'] for r in ledger})} repositories")


if __name__ == "__main__":
    main()
