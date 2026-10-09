"""Count OneJev downloads on Hugging Face and stars on GitHub.

    python scripts/stats.py

Appends today's download numbers to assets/downloads.csv and redraws
assets/downloads.svg, assets/stars.svg and their dark versions.
"""
import csv
import datetime
import json
import os
import urllib.request

API = "https://huggingface.co/api/models"
OWNER = "OmniJev"
FAMILY = "OmniJev/OneJev"
KINDS = ("quantized", "finetune", "adapter", "merge")
LEDGER = "assets/downloads.csv"
GITHUB = "OmniJev/OneJev"
W, H = 920, 300
PAD = (152, 26, 34, 52)
LIGHT = {"paper": "#fefefe", "ink": "#1e1e1e", "grid": "#dcdcdc", "muted": "#6b6b6b",
         "ours": "#d45bb6", "community": "#f9cadd"}
DARK = {"paper": "#1e1e1e", "ink": "#fefefe", "grid": "#3a3a3a", "muted": "#9a9a9a",
        "ours": "#d45bb6", "community": "#f386a1"}


def get(url):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    headers = {"User-Agent": "onejev-downloads"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return json.load(r)


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


def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def update(rows):
    old = read_ledger()
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
    dates = sorted({r["date"] for r in ledger})
    per_repo = {}
    for r in ledger:
        per_repo.setdefault(r["repo"], {})[r["date"]] = (r["side"], int(r["downloads"]))
    stack = []
    for d in dates:
        ours = community = 0
        for repo, points in per_repo.items():
            seen = [v for day, v in sorted(points.items()) if day <= d]
            if not seen:
                continue
            side, value = seen[-1]
            if side == "ours":
                ours += value
            else:
                community += value
        stack.append((d, ours + community))
    return stack


def human(n):
    if n >= 1000000:
        return f"{n / 1000000:.1f}M"
    if n >= 10000:
        return f"{n / 1000:.0f}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def ticks(top):
    step = 10 ** max(len(str(int(top))) - 2, 0)
    for mult in (1, 2, 2.5, 5, 10, 20, 25, 50):
        if top / (step * mult) <= 4:
            step = step * mult
            break
    out, v = [], 0.0
    while v < top + step:
        out.append(int(v))
        v += step
    return out


def stars():
    out, page = [], 1
    while True:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{GITHUB}/stargazers?per_page=100&page={page}",
            headers={"Accept": "application/vnd.github.star+json", "User-Agent": "onejev-stats"})
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=60) as r:
            batch = json.load(r)
        out += [s["starred_at"][:10] for s in batch]
        if len(batch) < 100:
            break
        page += 1
    out.sort()
    today = datetime.datetime.now(datetime.timezone.utc).date()
    day = datetime.date.fromisoformat(out[0]) if out else today
    series, i = [], 0
    while day <= today:
        while i < len(out) and out[i] <= day.isoformat():
            i += 1
        series.append((day.isoformat(), i))
        day += datetime.timedelta(days=1)
    return series


def chart(stack, c, title, unit):
    left, right, top, bottom = PAD
    x0, x1 = left, W - right
    y0, y1 = top, H - bottom
    bands = len(stack[0]) - 1
    days = [datetime.date.fromisoformat(r[0]) for r in stack]
    span = max((days[-1] - days[0]).days, 1)
    tops = ticks(max(sum(r[1:]) for r in stack) or 1)
    ymax = max(tops[-1], 1)
    px = lambda d: x0 + (d - days[0]).days / span * (x1 - x0)
    py = lambda v: y1 - v / ymax * (y1 - y0)
    total = sum(stack[-1][1:])
    parts = [f'<rect width="{W}" height="{H}" fill="{c["paper"]}"/>']
    for t in tops:
        y = py(t)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{c["grid"]}" stroke-width="1"/>')
        if t:
            parts.append(f'<text x="{x0 + 6}" y="{y - 6:.1f}" font-size="12" fill="{c["muted"]}">{human(t)}</text>')
    base = f"{px(days[-1]):.1f},{py(0):.1f} {px(days[0]):.1f},{py(0):.1f}"
    line = lambda pts: " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    fills = [c["community"], c["ours"]] if bands == 2 else [c["ours"]]
    for k in range(bands, 0, -1):
        pts = [(px(d), py(sum(r[1:k + 1]))) for d, r in zip(days, stack)]
        parts.append(f'<polygon points="{line(pts)} {base}" fill="{fills[bands - k]}"/>')
    top_pts = [(px(d), py(sum(r[1:]))) for d, r in zip(days, stack)]
    parts.append(f'<polyline points="{line(top_pts)}" fill="none" stroke="{c["ink"]}" stroke-width="2"/>')
    if len(days) < 40:
        for x, y in top_pts:
            parts.append(f'<rect x="{x - 2.5:.1f}" y="{y - 2.5:.1f}" width="5" height="5" fill="{c["ink"]}"/>')
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{c["ink"]}" stroke-width="2"/>')
    for d in (days[0], days[-1]):
        anchor = "start" if d == days[0] else "end"
        parts.append(f'<text x="{px(d):.1f}" y="{y1 + 20}" text-anchor="{anchor}" font-size="12" fill="{c["muted"]}">{d.isoformat()}</text>')
    parts.append(f'<text x="{x0 - 140}" y="{top + 26}" font-size="32" font-weight="700" fill="{c["ink"]}">{total:,}</text>')
    parts.append(f'<text x="{x0 - 140}" y="{top + 46}" font-size="12" fill="{c["muted"]}">{unit}</text>')
    if bands == 2:
        parts.append(f'<rect x="{x0 - 140}" y="{top + 72}" width="10" height="10" fill="{c["ours"]}"/>')
        parts.append(f'<text x="{x0 - 124}" y="{top + 81}" font-size="12" fill="{c["ink"]}">OneJev {human(stack[-1][1])}</text>')
        parts.append(f'<rect x="{x0 - 140}" y="{top + 92}" width="10" height="10" fill="{c["community"]}"/>')
        parts.append(f'<text x="{x0 - 124}" y="{top + 101}" font-size="12" fill="{c["ink"]}">community {human(stack[-1][2])}</text>')
    else:
        parts.append(f'<text x="{x0 - 140}" y="{top + 72}" font-size="12" fill="{c["muted"]}">{title}</text>')
    body = "\n".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace">\n{body}\n</svg>\n')


def main():
    ledger = update(family())
    stack = series(ledger)
    for name, colours in (("assets/downloads.svg", LIGHT), ("assets/downloads-dark.svg", DARK)):
        open(name, "w", encoding="utf-8").write(chart(stack, colours, "Hugging Face", "downloads"))
    print(f"{stack[-1][0]}: {stack[-1][1]} downloads over "
          f"{len({r['repo'] for r in ledger})} repositories")
    counts = stars()
    for name, colours in (("assets/stars.svg", LIGHT), ("assets/stars-dark.svg", DARK)):
        open(name, "w", encoding="utf-8").write(chart(counts, colours, GITHUB, "stars"))
    print(f"{counts[-1][0]}: {counts[-1][1]} stars on {GITHUB}")


if __name__ == "__main__":
    main()
