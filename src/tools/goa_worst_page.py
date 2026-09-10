#!/usr/bin/env python3
"""goa_worst_page.py -- the N worst-bandwidth tracks in the goa archive, as a playlist
plus a browsable page.

Kim 2026-08-21: "make me a link list to 100 worst big goa tracks, I'm curious, mp3
encoding alone should not collapse the quality that bad." He is right, and the page
says so: 76 of the worst 100 declare >=192 kbps while their content stops between 4.7
and 11.9 kHz -- below what even a 128k encode produces. The header records the last
re-encode, not the damage.

LAYOUT NOTE (v2): the first version put the full relative path in one column clamped
with `max-width:0; text-overflow:ellipsis`, which renders as an ellipsis and identifies
nothing -- useless for the one job the page has. Path is now split into ALBUM and TRACK
columns that wrap, with the full path on hover and the file:// link on the track name.

The .m3u is the primary artifact: open it in any player and audition straight through.
The .html is for reading the numbers; its file:// links only work when the page itself
is opened locally (a browser blocks file:// navigation from an https:// page).

RUN (mir venv, seconds):
  mir/bin/python src/tools/goa_worst_page.py --features <UUID>/goa_archive_features \\
      --out stats/goa_big_worst100 [--n 100]
"""
import argparse
import html
import json
import os
import re
from pathlib import Path

import numpy as np

ARCHIVE_ROOT = "/run/media/kim/Mantu/goa_archive_extracted/"
ROLE_SHORT = {"unique": "uniq", "duplicate_dropped": "dup&#10007;",
              "duplicate_best": "dup&#9733;", "master_variant": "var"}

CSS = """
:root{--bg:#14161a;--card:#1b1e23;--ink:#e6e8e4;--dim:#9aa0a8;--hair:#262b31;
      --bad:#e08a80;--accent:#8fc4ee;--warn:#d8ae55}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,sans-serif;
     margin:0;padding:24px 22px 64px}
h1{font-size:20px;margin:0 0 5px}
p.sub{color:var(--dim);margin:0 0 16px;max-width:88ch}
.box{background:var(--card);border-left:3px solid var(--warn);padding:12px 16px;
     margin:0 0 18px;max-width:100ch;border-radius:0 3px 3px 0}
b{color:#f0d79a}
.key{color:var(--dim);font-size:12.5px}
table{border-collapse:collapse;width:100%;table-layout:auto;
      font:12.5px/1.4 ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
th{position:sticky;top:0;background:var(--bg);color:var(--accent);text-align:left;
   padding:7px 8px;border-bottom:1px solid var(--hair);font-size:10.5px;
   text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
td{padding:6px 8px;border-bottom:1px solid #20242a;vertical-align:top}
/* numeric columns shrink to content; the two TEXT columns get the rest and WRAP,
   which is the whole point of v2 -- nothing here may be clipped to an ellipsis. */
td.n{text-align:right;white-space:nowrap;width:1%}
td.bad{color:var(--bad);font-weight:600}
td.yr{color:var(--dim)}
td.ro{white-space:nowrap;width:1%;color:var(--dim);font-size:11px}
td.ro.r-dup{color:#767d85}
td.al{color:var(--dim);word-break:break-word;min-width:18ch}
td.tk{word-break:break-word;min-width:24ch}
td.tk a{color:var(--accent);text-decoration:none}
td.tk a:hover{text-decoration:underline}
tr:hover td{background:#1a1e24}
/* narrow screens: drop the album column rather than squeeze the track name */
@media (max-width:860px){td.al,th.al{display:none}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="goa_archive_features dir (holds quality/quality.jsonl + curated.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    feats, out = Path(a.features), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows_q = []
    with (feats / "quality" / "quality.jsonl").open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("cutoff_hz") is not None:
                rows_q.append(r)
    rows_q.sort(key=lambda r: r["cutoff_hz"])
    worst = rows_q[:a.n]

    role = {}
    cur = feats / "curated.jsonl"
    if cur.exists():
        with cur.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                role[d.get("rel")] = d.get("role")

    def rel(r):
        p = r["path"]
        return p[len(ARCHIVE_ROOT):] if p.startswith(ARCHIVE_ROOT) else p

    # ---- playlist: the primary artifact ----
    with (out / "worst100.m3u").open("w") as f:
        f.write("#EXTM3U\n")
        for r in worst:
            f.write("#EXTINF:%d,%.1f kHz | %sk hdr | %s\n"
                    % (int(r.get("dur") or 0), r["cutoff_hz"] / 1000,
                       r.get("bitrate_k") or "?", os.path.basename(rel(r))))
            f.write(r["path"] + "\n")
    with (out / "worst100.txt").open("w") as f:
        for r in worst:
            f.write(r["path"] + "\n")

    # ---- page ----
    trs = []
    for i, r in enumerate(worst, 1):
        rp = rel(r)
        parts = rp.split("/")
        album = parts[1] if len(parts) > 2 else (parts[0] if len(parts) > 1 else "")
        track = os.path.splitext(parts[-1])[0]
        m = re.search(r"(19|20)\d{2}", parts[0]) or re.search(r"(19|20)\d{2}", album)
        yr = m.group(0) if m else "&middot;"
        raw = role.get(rp, "")
        ro = ROLE_SHORT.get(raw, "?")
        cls = " r-dup" if raw == "duplicate_dropped" else ""
        href = html.escape(r["path"].replace('"', "%22"))
        trs.append(
            "<tr><td class=n>%d</td>"
            '<td class="n bad">%.1f</td>'
            "<td class=n>%s</td><td class=n>%d</td>"
            '<td class="n yr">%s</td>'
            '<td class="ro%s">%s</td>'
            "<td class=al>%s</td>"
            '<td class=tk><a href="file://%s" title="%s">%s</a></td></tr>'
            % (i, r["cutoff_hz"] / 1000, r.get("bitrate_k") or "?", int(r.get("dur") or 0),
               yr, cls, ro, html.escape(album), href, html.escape(rp), html.escape(track)))

    br = [r["bitrate_k"] for r in worst if r.get("bitrate_k")]
    hi192 = sum(1 for r in worst if (r.get("bitrate_k") or 0) >= 192)
    hi256 = sum(1 for r in worst if (r.get("bitrate_k") or 0) >= 256)
    lo128 = sum(1 for r in worst if (r.get("bitrate_k") or 0) <= 128)
    ndup = sum(1 for r in worst if role.get(rel(r)) == "duplicate_dropped")
    nalb = len({"/".join(rel(r).split("/")[:2]) for r in worst})
    lo = worst[0]["cutoff_hz"] / 1000
    hi = worst[-1]["cutoff_hz"] / 1000
    q = np.percentile(br, [25, 50, 75]) if br else [0, 0, 0]

    doc = (
        "<!doctype html><meta charset=utf-8>"
        "<title>Worst %d big-goa tracks</title>\n<style>%s</style>\n"
        "<h1>Worst %d tracks in the goa archive</h1>\n"
        "<p class=sub>Ranked by spectral cutoff &mdash; the real bandwidth present in the "
        "audio, not what the header claims. Click a track to play it; hover for the full path.</p>\n"
        "<div class=box><b>These are not just low-bitrate MP3s.</b> All %d are mp3, but "
        "<b>%d declare &ge;192&nbsp;kbps</b> and <b>%d declare &ge;256&nbsp;kbps</b>, while their "
        "actual content stops between <b>%.1f and %.1f&nbsp;kHz</b>. A clean 192k encode reaches "
        "~18&ndash;19&nbsp;kHz; even 128k reaches ~16. Content ending at 8&nbsp;kHz from a file "
        "claiming 320k means the <b>source was already destroyed before this encode</b> &mdash; a "
        "transcode of a transcode, an analog/tape rip, or a stream capture. Only %d have a genuinely "
        "low header (&le;128k). Spread across <b>%d distinct albums</b>, so it is scattered vintage "
        "sourcing, not one bad rip batch.<br><br>"
        "<span class=key>role: <b>uniq</b> = only copy &middot; <b>dup&#9733;</b> = best of its "
        "duplicate set &middot; <b>dup&#10007;</b> = a better copy exists elsewhere in the archive "
        "&middot; <b>var</b> = alternate mastering (kept). %d of these already have a better sibling."
        "</span></div>\n"
        "<p class=sub>Header bitrate p25/median/p75 = %d/%d/%d kbps. Playlist beside this file: "
        "<code>worst100.m3u</code>.</p>\n"
        "<table><thead><tr><th class=n>#</th><th class=n>kHz</th><th class=n>hdr</th>"
        "<th class=n>sec</th><th class=n>yr</th><th>role</th><th class=al>album</th>"
        "<th>track</th></tr></thead>\n<tbody>\n%s\n</tbody></table>\n"
        % (a.n, CSS, a.n, len(worst), hi192, hi256, lo, hi, lo128, nalb, ndup,
           q[0], q[1], q[2], "\n".join(trs)))
    (out / "worst100.html").write_text(doc)
    print("wrote %s/{worst100.m3u,worst100.html,worst100.txt}" % out)
    print("cutoff range %.2f-%.2f kHz across %d albums; %d already have a better sibling"
          % (lo, hi, nalb, ndup))


if __name__ == "__main__":
    main()
