(() => {
  "use strict";
  const LANG_KEY = "onejev-lang";
  const LANG = (() => {
    const q = new URLSearchParams(location.search).get("lang");
    if (q === "zh" || q === "en") {
      try {
        localStorage.setItem(LANG_KEY, q);
      } catch (e) {
      }
      return q;
    }
    try {
      const v = localStorage.getItem(LANG_KEY);
      if (v === "zh" || v === "en") return v;
    } catch (e) {
    }
    return /^zh/i.test(navigator.language || "") ? "zh" : "en";
  })();
  const ZH = LANG === "zh";
  const say = (en, zh) => ZH ? zh : en;
  const MAIN = "OneJev-27B", BASE = "Qwen3.8-27B", SMALL = "OneJev-4B";
  const disp = (n) => String(n).replace(/OmniJev/g, "OneJev");
  const MAIN_D = disp(MAIN), SMALL_D = disp(SMALL);
  const COMPARE = [[MAIN, MAIN_D], [BASE, say("Qwen3.8-27B before training", "Qwen3.8-27B（训练前）")], [SMALL, SMALL_D]];
  const GROUPS = [["gui", say("GUI agents", "GUI agent")], ["image", say("Images", "图片")], ["video_long", say("Long video", "长视频")], ["video_short", say("Short video", "短视频")], ["text", say("Text", "文本")]];
  const $ = (s, r = document) => r.querySelector(s);
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "style") e.style.cssText = v;
      else if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const k of kids.flat(3)) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
    return e;
  }
  function win(title, body, o = {}) {
    return h(
      "div",
      { class: "win " + (o.cls || ""), style: o.style },
      h("div", { class: "win-t" }, h("span", { class: "t" }, title), o.bar || null),
      h("div", { class: "win-b " + (o.bcls || "") }, body)
    );
  }
  const f3 = (x) => x == null ? "" : x.toFixed(3);
  const f1 = (x) => x == null ? "" : x.toFixed(1);
  const p2 = (x) => x.toFixed(2);
  const int = (n) => Number(n).toLocaleString("en-US");
  const pct = (x) => (x * 100).toFixed(1);
  const short = (s, n) => s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s;
  const top = (pr) => Object.entries(pr).reduce((a, b) => b[1] > a[1] ? b : a);
  const optText = (q, k) => (q.question.options.find((o) => o.key === k) || { text: k }).text;
  const dur = (s) => ZH ? String(s).replace(/ min /g, " 分 ").replace(/ s$/, " 秒") : s;
  const elText = (s) => s.replace(/,? center at about .*$/, "").replace(/ \[\.\.\. \d+ chars cut \.\.\.\] /, "…");
  function mulberry(a) {
    return () => {
      a |= 0;
      a = a + 1831565813 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  const BAYER = (() => {
    const m = [[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22], [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25], [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]];
    return m.map((r) => r.map((v) => (v + 0.5) / 64));
  })();
  function drawField(canvas, seed) {
    const cell = 4, r = canvas.getBoundingClientRect();
    const W = Math.max(1, Math.ceil(r.width / cell)), H = Math.max(1, Math.ceil(r.height / cell));
    canvas.width = W;
    canvas.height = H;
    const rnd = mulberry(seed);
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(W, H), d = img.data;
    const hills = [{ x: 0.18, w: 0.34, h: 0.3 }, { x: 0.62, w: 0.46, h: 0.22 }, { x: 0.95, w: 0.28, h: 0.36 }];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const u = x / W, v = y / H;
      let top2 = 1;
      for (const hl of hills) {
        const dx = (u - hl.x) / hl.w;
        top2 = Math.min(top2, 1 - hl.h * Math.max(0, 1 - dx * dx));
      }
      let dens = Math.max(0, (v - (top2 - 0.16)) / 0.34);
      dens = Math.min(1, dens) * 0.9 + (rnd() - 0.5) * 0.04;
      if (dens > BAYER[y & 7][x & 7]) {
        const i = (y * W + x) * 4, deep = v > top2 + 0.06;
        d[i] = deep ? 212 : 243;
        d[i + 1] = deep ? 91 : 134;
        d[i + 2] = deep ? 182 : 161;
        d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  const EYE = [
    "################",
    "#..............#",
    "#....mmmmmm....#",
    "#..mm......mm..#",
    "#.m...oooo...m.#",
    "#m...oooooo...m#",
    "#.m...oooo...m.#",
    "#..mm......mm..#",
    "#....mmmmmm....#",
    "#..............#",
    "################",
    "......####......"
  ];
  const EYE_IN = { 3: [5, 10], 4: [3, 12], 5: [2, 13], 6: [3, 12], 7: [5, 10] };
  const PUPIL = [[4, 6, 9], [5, 5, 10], [6, 6, 9]];
  const eyes = [];
  const SVGNS = "http://www.w3.org/2000/svg";
  function sv(tag, a) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in a) e.setAttribute(k, a[k]);
    return e;
  }
  function eyeSVG(host) {
    const svg2 = sv("svg", { viewBox: "0 0 16 12", class: "eye-svg", "shape-rendering": "crispEdges", role: "img", "aria-label": "OneJev eye" });
    const frame = sv("g", { fill: "#1e1e1e" }), ring = sv("g", { fill: "#d45bb6" }), pupil = sv("g", { fill: "#f386a1" }), lid = sv("g", { fill: "#d45bb6" });
    EYE.forEach((row, y) => [...row].forEach((ch, x) => {
      if (ch === "#") frame.append(sv("rect", { x, y, width: 1, height: 1 }));
      if (ch === "m") ring.append(sv("rect", { x, y, width: 1, height: 1 }));
    }));
    lid.append(sv("rect", { x: 1, y: 5, width: 14, height: 1 }));
    lid.style.display = "none";
    svg2.append(frame, ring, pupil, lid);
    host.append(svg2);
    const e = { svg: svg2, ring, pupil, lid, dx: 0, dy: 0 };
    e.look = (dx, dy) => {
      if (dx === e.dx && dy === e.dy && pupil.childNodes.length) return;
      e.dx = dx;
      e.dy = dy;
      pupil.replaceChildren();
      for (const [y0, a, b] of PUPIL) {
        const y = y0 + dy, lim = EYE_IN[y];
        if (!lim) continue;
        for (let x = a + dx; x <= b + dx; x++) if (x >= lim[0] && x <= lim[1]) pupil.append(sv("rect", { x, y, width: 1, height: 1 }));
      }
      const hx = PUPIL[0][1] + dx, hy = PUPIL[0][0] + dy;
      if (EYE_IN[hy] && hx >= EYE_IN[hy][0]) pupil.append(sv("rect", { x: hx, y: hy, width: 1, height: 1, fill: "#fefefe" }));
    };
    e.shut = (closed) => {
      ring.style.display = pupil.style.display = closed ? "none" : "";
      lid.style.display = closed ? "" : "none";
    };
    e.look(0, 0);
    eyes.push(e);
    return e;
  }
  function eyesInit() {
    [["#nav-eye"], ["#logo-eye"], ["#foot-eye"]].forEach(([sel]) => {
      const hst = $(sel);
      if (hst) eyeSVG(hst);
    });
    let raf = 0, px = innerWidth / 2, py = innerHeight / 2;
    const aim = () => {
      raf = 0;
      for (const e of eyes) {
        const r = e.svg.getBoundingClientRect();
        if (!r.width) continue;
        const cx = r.left + r.width / 2, cy = r.top + r.height * 0.42, dx = px - cx, dy = py - cy, d = Math.hypot(dx, dy) || 1;
        const reach = Math.min(1, d / (r.width * 1.5));
        e.look(Math.round(dx / d * 2 * reach), Math.round(dy / d * 1.2 * reach));
      }
    };
    addEventListener("pointermove", (ev) => {
      px = ev.clientX;
      py = ev.clientY;
      if (!raf) raf = requestAnimationFrame(aim);
    });
    addEventListener("scroll", () => {
      if (!raf) raf = requestAnimationFrame(aim);
    }, { passive: true });
    const blink = () => {
      if (!document.hidden) {
        eyes.forEach((e) => e.shut(true));
        setTimeout(() => eyes.forEach((e) => e.shut(false)), 150);
      }
      setTimeout(blink, 4200 + Math.random() * 3800);
    };
    setTimeout(blink, 3e3);
    const sub = $("#banner-sub");
    if (sub) {
      const full = sub.textContent;
      let i = 0;
      sub.textContent = "";
      const cur = h("span", { class: "cursor" });
      const tick = () => {
        i++;
        sub.textContent = full.slice(0, i);
        sub.append(cur);
        if (i < full.length) setTimeout(tick, 38);
      };
      setTimeout(tick, 250);
    }
  }
  function blocks(v, max, cls, label) {
    const el = h("div", { class: `bb ${cls || ""} wait` });
    if (v == null || !isFinite(v)) {
      el.append(h("span", { class: "bn" }, label || "n/a"));
      return el;
    }
    const n = Math.max(1, Math.min(28, Math.round(v / max * 28)));
    for (let k = 0; k < n; k++) el.append(h("i", { style: `--k:${k}` }));
    if (label !== "") el.append(h("span", { class: "bv", style: `grid-column:${n + 1} / span 6` }, label ?? v.toFixed(1)));
    return el;
  }
  function buildOnView(root) {
    visible(root, (v) => {
      if (!v) return;
      root.querySelectorAll(".bb.wait").forEach((b) => {
        const is = [...b.querySelectorAll("i")];
        b.classList.remove("wait");
        is.forEach((i) => {
          i.style.visibility = "hidden";
        });
        is.forEach((i, k) => setTimeout(() => {
          i.style.visibility = "visible";
        }, 60 + k * 26));
      });
      root.querySelectorAll("table.ptable").forEach((t) => t.classList.add("go"));
    }, 0.15);
  }
  function loadImg(src) {
    return new Promise((ok, no) => {
      const i = new Image();
      i.onload = () => ok(i);
      i.onerror = no;
      i.src = src;
    });
  }
  function atkinson(img, W, light = [249, 202, 221], noise = 0) {
    const H = Math.max(1, Math.round(img.naturalHeight * W / img.naturalWidth));
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const x = c.getContext("2d");
    x.drawImage(img, 0, 0, W, H);
    const id = x.getImageData(0, 0, W, H), d = id.data, g = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) g[i] = Math.min(255, Math.max(0, (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2] - 128) * 1.25 + 140 + (noise ? (Math.random() - 0.5) * noise : 0)));
    const add = (xx, yy, e) => {
      if (xx >= 0 && xx < W && yy < H) g[yy * W + xx] += e;
    };
    for (let yy = 0; yy < H; yy++) for (let xx = 0; xx < W; xx++) {
      const i = yy * W + xx, old = g[i], nw = old < 128 ? 0 : 255, e = (old - nw) / 8;
      add(xx + 1, yy, e);
      add(xx + 2, yy, e);
      add(xx - 1, yy + 1, e);
      add(xx, yy + 1, e);
      add(xx + 1, yy + 1, e);
      add(xx, yy + 2, e);
      const [r, gg, b] = nw ? light : [30, 30, 30];
      d[4 * i] = r;
      d[4 * i + 1] = gg;
      d[4 * i + 2] = b;
      d[4 * i + 3] = 255;
    }
    x.putImageData(id, 0, 0);
    c.style.imageRendering = "pixelated";
    c.style.width = "100%";
    return c;
  }
  function draggable(w, box) {
    const bar = w.querySelector(".win-t");
    w.classList.add("grab");
    bar.addEventListener("pointerdown", (ev) => {
      if (window.innerWidth < 900 || ev.target.closest("button")) return;
      const sx = ev.clientX, sy = ev.clientY, ox = w.offsetLeft, oy = w.offsetTop;
      document.querySelectorAll(".os .win").forEach((o) => o.classList.remove("lift"));
      w.classList.add("lift");
      bar.setPointerCapture(ev.pointerId);
      const mv = (e) => {
        w.style.left = Math.max(-w.offsetWidth + 60, Math.min(box.clientWidth - 60, ox + e.clientX - sx)) + "px";
        w.style.top = Math.max(0, Math.min(box.clientHeight - 30, oy + e.clientY - sy)) + "px";
      };
      const up = () => {
        bar.removeEventListener("pointermove", mv);
        bar.removeEventListener("pointerup", up);
      };
      bar.addEventListener("pointermove", mv);
      bar.addEventListener("pointerup", up);
    });
  }
  function visible(el, cb, threshold = 0.2) {
    const io = new IntersectionObserver((es) => cb(es[es.length - 1].isIntersecting), { threshold });
    io.observe(el);
  }
  function tabs(box, labels, onPick, auto = 0) {
    box.innerHTML = "";
    let cur = 0, user = false, seen = false;
    const bs = labels.map((l, i) => h("button", { type: "button", "aria-pressed": i === 0 ? "true" : "false", onclick: (ev) => pick(i, ev.isTrusted) }, l));
    function pick(i, byUser) {
      cur = i;
      if (byUser) user = true;
      bs.forEach((b, j) => b.setAttribute("aria-pressed", i === j ? "true" : "false"));
      onPick(i);
    }
    box.append(...bs);
    onPick(0);
    if (auto && labels.length > 1) {
      visible(box.closest(".desk") || box, (v) => {
        seen = v;
      });
      setInterval(() => {
        if (!user && seen && !document.hidden) pick((cur + 1) % labels.length, false);
      }, auto);
    }
  }
  function grow(root) {
    const bars = [...root.querySelectorAll(".bar i, .steplist .b i")], gs = [...root.querySelectorAll(".gauge div i")];
    bars.forEach((e) => {
      e.dataset.w = e.style.width;
      e.style.transition = "none";
      e.style.width = "0%";
    });
    gs.forEach((e) => {
      e.dataset.h = e.style.height;
      e.style.transition = "none";
      e.style.height = "0%";
    });
    requestAnimationFrame(() => requestAnimationFrame(() => {
      bars.forEach((e) => {
        e.style.transition = "";
        e.style.width = e.dataset.w;
      });
      gs.forEach((e) => {
        e.style.transition = "";
        e.style.height = e.dataset.h;
      });
    }));
  }
  function player(frames, o = {}) {
    let i = 0, on = true, t = null;
    const fps = o.fps || 5;
    const imgs = frames.map((src) => {
      const im = new Image();
      im.src = src;
      im.alt = "";
      return im;
    });
    const screen = h("div", { class: "screen" }, imgs[0]);
    const range = h("input", { type: "range", min: 0, max: frames.length - 1, value: 0, "aria-label": say("frame", "帧") });
    const count = h("span", {}, `1/${frames.length}`);
    const btn = h("button", { type: "button" }, say("stop", "暂停"));
    function show(k) {
      i = k;
      screen.replaceChildren(imgs[i]);
      range.value = i;
      count.textContent = `${i + 1}/${frames.length}`;
    }
    function tick() {
      if (!on) return;
      if (i === frames.length - 1) {
        clearInterval(t);
        t = setTimeout(() => {
          show(0);
          run();
        }, 1400);
        return;
      }
      show(i + 1);
    }
    function run() {
      clearInterval(t);
      clearTimeout(t);
      t = setInterval(tick, 1e3 / fps);
    }
    btn.onclick = () => {
      on = !on;
      btn.textContent = on ? say("stop", "暂停") : say("play", "播放");
      if (on) run();
    };
    range.oninput = () => {
      on = false;
      btn.textContent = say("play", "播放");
      show(+range.value);
    };
    const el = h("div", { class: "player" }, screen, h("div", { class: "ctrl" }, btn, range, count), o.extra || null);
    run();
    return { el, stop() {
      on = false;
      clearInterval(t);
      clearTimeout(t);
    } };
  }
  function answerBars(q, models = COMPARE) {
    const box = h("div", { class: "bars" });
    for (const [name, label] of models) {
      const pr = q.probs[name];
      if (!pr) continue;
      const [k, p] = top(pr), ok = k === q.gold;
      box.append(h(
        "div",
        { class: "bar-row" },
        h("div", { class: "who" }, h("span", {}, label), h("span", { class: ok ? "ok" : "no" }, ok ? say("right", "对") : say("wrong", "错"))),
        h("div", { class: "pick" }, short(elText(optText(q, k)), 110)),
        h("div", { class: "v" }, p2(p)),
        h("div", { class: "bar " + (name === MAIN ? "" : name === BASE ? "dither" : "light") }, h("i", { style: `width:${(p * 100).toFixed(1)}%` }))
      ));
    }
    return box;
  }
  function nOptions(q) {
    const n = q.question.n_options;
    return q.question.type === "noul" ? say("yes or no", "是或否") : say(`one of ${n} options`, `${n} 选 1`);
  }
  function typeInto(pre, text, speed = 4) {
    if (pre._timer) cancelAnimationFrame(pre._timer);
    let i = 0;
    const step = () => {
      i = Math.min(text.length, i + speed);
      pre.textContent = text.slice(0, i);
      pre.append(h("span", { class: "cur" }));
      if (i < text.length) pre._timer = requestAnimationFrame(step);
    };
    step();
  }
  function answerJSON(q, name = MAIN) {
    const pr = q.probs[name], t = q.question.type, key = q.task;
    if (t === "noul") return `"${key}": {"type": "noul", "noul": ${pr.yes.toFixed(4)}}`;
    const [k, p] = top(pr);
    if (t === "choice") return `"${key}": {"type": "choice", "choice": "${short(k, 40)}", "confidence": ${p.toFixed(4)}, "probabilities": {...}}`;
    const score = Object.entries(pr).reduce((s, [kk, v]) => s + Number(kk) * v, 0);
    return `"${key}": {"type": "score", "score": ${score.toFixed(2)}, "probabilities": {...}}`;
  }
  function requestText(e, n) {
    const st = e.state, keep = ["platform", "task", "website", "recipe", "elapsed"];
    const lines = [];
    for (const k of keep) if (st[k] != null) lines.push(`    "${k}": ${JSON.stringify(short(String(st[k]), 90))}`);
    if (st.steps_so_far != null) lines.push(`    "steps_so_far": ${st.steps_so_far}`);
    if (st.total_steps != null) lines.push(`    "total_steps": ${st.total_steps}`);
    if (st.steps) lines.push(`    "steps": [${st.steps.length} steps shown]`);
    if (st.previous_actions) lines.push(`    "previous_actions": [${st.previous_actions.length} actions]`);
    for (const [k, v] of Object.entries(st)) if (typeof v === "string" && /^<(image|video):\d+>$/.test(v)) lines.push(`    "${k}": "${v}"`);
    const qs = e.questions.map((q) => `    "${q.task}": {"type": "${q.question.type}", "instructions": ${JSON.stringify(short(q.question.text, 70))}}`);
    return `$ curl -s localhost:8000/v1/systemone -d @request.json
{
  "model": "${n}",
  "state": {
${lines.join(",\n")}
  },
  "media": [${e.media.length} ${e.media[0].type === "video" ? "video" : "image"}${e.media.length > 1 ? "s" : ""}],
  "questions": {
${qs.join(",\n")}
  }
}

{"model": "${n}", "answers": {
  ${e.questions.map((q) => answerJSON(q)).join(",\n  ")}
}}
`;
  }
  const WEB_RUN = say("web agent run", "网页 agent 运行"), DESK_RUN = say("desktop agent run", "桌面 agent 运行"), PHONE_RUN = say("phone agent run", "手机 agent 运行");
  const GUI_SOURCES = { agent_reward_bench: WEB_RUN, openhands_webarena: WEB_RUN, multimodal_mind2web: say("web page", "网页"), mlfoundations_osworld_trajs: DESK_RUN, osworld_verified_trajs: DESK_RUN, mobileworld_progrm_rollouts: PHONE_RUN, amex: say("phone screen", "手机屏幕"), misactbench: DESK_RUN, seerray_androidworld_eval: PHONE_RUN, cua_speedrun_trajectories: DESK_RUN };
  const srcName = (s) => GUI_SOURCES[s] || s.replace(/^cauldron_/, "").replace(/_/g, " ");
  const STREAMS = [
    ["noul", say("filter", "筛选"), say("yes or no", "是或否"), say("yes-or-no", "是非")],
    ["choice", say("decide", "决策"), say("one of many", "多选一"), say("multiple-choice", "多选")],
    ["score", say("rate", "评分"), say("on a scale", "按等级"), say("rating", "评分")]
  ];
  const LEVEL_OFF = ZH ? ["完全一致", "差一级", "差两级", "差三级", "差四级"] : ["exact", "one level off", "two levels off", "three levels off", "four levels off"];
  let decisions = 0, osSeen = false;
  function streamWindow(kind, label, sub, noun, items, S) {
    items = items.filter((x) => x.source !== "hateful_memes");
    if (!items.length) return null;
    const img = h("img", { alt: "" }), dith = h("div", { class: "st-dith" }), stamp = h("div", { class: "st-stamp" }), src = h("span", { class: "st-src" });
    const stage = h("div", { class: "st-stage" }, img, dith, stamp, src);
    const ctx = h("p", { class: "st-ctx" }), q = h("p", { class: "st-q" });
    const rows = [], body = h("div", { class: "st-opts" });
    let meter = null;
    if (kind === "score") {
      const cols = [...Array(5)].map((_, j) => {
        const m = h("i", { class: "m" }), b = h("i", { class: "b" }), n = h("span", { class: "sm-n" }, j + 1);
        const col = h("div", { class: "sm-col" }, h("div", { class: "sm-bar" }, b, m), n);
        return { col, m, b, n };
      });
      const tm = h("i", { class: "m" }), tb = h("i", { class: "b" });
      const ans = h("p", { class: "sm-ans" });
      meter = { cols, tm, tb, ans, box: h("div", { class: "sm" }, cols.map((c) => c.col)) };
      body.append(ans, meter.box, h("div", { class: "so-track sm-track" }, tb, tm), h("p", { class: "st-legend" }, "■ ", MAIN_D, say(" expected level   □ ", " 期望等级   □ "), BASE, say(" before training", " 训练前")));
    } else {
      if (kind === "noul") {
        meter = { big: h("p", { class: "st-big" }, h("span", { class: "w" }), h("span", { class: "p" })) };
        body.append(meter.big);
      }
      for (let j = 0; j < (kind === "noul" ? 2 : 5); j++) {
        const l = h("span", { class: "so-l" }), v = h("span", { class: "so-v" }), m = h("i", { class: "m" }), b = h("i", { class: "b" });
        const row = h("div", { class: "so-row" }, h("div", { class: "so-top" }, l, v), h("div", { class: "so-track" }, b, m));
        rows.push({ row, l, v, m, b });
        body.append(row);
      }
      body.append(h("p", { class: "st-legend" }, "■ ", MAIN_D, "   □ ", BASE, say(" before training", " 训练前")));
    }
    const verdict = h("p", { class: "st-verdict" });
    const hist = h("div", { class: "st-hist" });
    const count = h("span", { class: "st-count" });
    const byT = S.by_type && S.by_type[`${kind}_media`];
    let foot = null;
    if (byT) {
      const a = byT.acc[MAIN], b = byT.acc[BASE];
      foot = kind === "score" ? say(`Over all ${int(byT.n)} ${noun} questions with media, ${MAIN_D} picks the exact level ${pct(a)} percent of the time and lands within one level ${pct(byT.within1[MAIN])} percent. ${BASE} before training gets ${pct(b)} and ${pct(byT.within1[BASE])} percent.`, `在全部 ${int(byT.n)} 道带媒体的${noun}题上，${MAIN_D} 选中准确等级的比例是 ${pct(a)}%，差一级以内的是 ${pct(byT.within1[MAIN])}%。训练前的 ${BASE} 分别是 ${pct(b)}% 和 ${pct(byT.within1[BASE])}%。`) : say(`Over all ${int(byT.n)} ${noun} questions with media, ${MAIN_D} is right ${pct(a)} percent of the time and ${BASE} before training ${pct(b)} percent.`, `在全部 ${int(byT.n)} 道带媒体的${noun}题上，${MAIN_D} 的正确率是 ${pct(a)}%，训练前的 ${BASE} 是 ${pct(b)}%。`);
    }
    const w = win(
      say(`${label}, ${sub}`, `${label}，${sub}`),
      [stage, h("div", { class: "st-text" }, ctx, q, body, verdict), hist, foot ? h("p", { class: "st-foot" }, foot) : null],
      { cls: `stream k-${kind}`, bcls: "flush", bar: count }
    );
    const cache = new Map();
    const prep = (i2) => {
      if (!cache.has(i2)) {
        const it = items[i2], srcs = it.media.type === "video" ? it.media.frames : [it.media.src];
        cache.set(i2, Promise.all(srcs.map(loadImg)).then((ims) => ({ ims, dz: atkinson(ims[0], 180) })).catch(() => null));
        if (cache.size > 8) cache.delete(cache.keys().next().value);
      }
      return cache.get(i2);
    };
    let i = 0, paused = false, pinned = false, inView = false, timers = [], waiting = false;
    const later = (f, t) => timers.push(setTimeout(f, t));
    w.addEventListener("mouseenter", () => {
      paused = true;
    });
    w.addEventListener("mouseleave", () => {
      paused = false;
      if (waiting && !pinned) next();
    });
    stage.addEventListener("click", () => {
      pinned = !pinned;
      paused = pinned;
      w.classList.toggle("pinned", pinned);
      if (!pinned && waiting) next();
    });
    visible(w, (v) => {
      inView = v;
      if (v && waiting && !paused) next();
    }, 0.25);
    document.addEventListener("omni-os", () => {
      if (osSeen && waiting && !paused) next();
    });
    function mark(it) {
      if (kind === "score") {
        const top_ = it.opts.reduce((a, o) => o.m > a.m ? o : a), dist = Math.abs(+top_.k - +it.gold);
        return { ok: dist === 0, near: dist === 1, dist, top: top_ };
      }
      return { ok: !!it.ok.m };
    }
    async function show(k) {
      const it = items[k], pr = await prep(k);
      prep((k + 1) % items.length);
      if (!pr) {
        i = (k + 1) % items.length;
        later(next, 50);
        return;
      }
      const video = it.media.type === "video";
      count.textContent = `${k + 1}/${items.length}`;
      stamp.className = "st-stamp";
      stamp.textContent = "";
      src.textContent = video ? say(`${srcName(it.source)}, ${pr.ims.length} frames`, `${srcName(it.source)}，${pr.ims.length} 帧`) : srcName(it.source);
      dith.replaceChildren(pr.dz);
      dith.style.display = "block";
      img.style.opacity = "0";
      stage.classList.remove("flash");
      void stage.offsetWidth;
      stage.classList.add("flash");
      const c = it.state.task || it.state.prompt || it.state.recipe || "";
      const cl = it.state.task ? say("Task. ", "任务：") : it.state.prompt ? say("Prompt. ", "提示：") : it.state.recipe ? say("Recipe. ", "菜谱：") : "";
      ctx.replaceChildren(c ? h("b", {}, cl) : "", c ? short(c, 150) : "");
      q.textContent = it.q;
      verdict.textContent = "";
      if (kind === "score") {
        const n = it.opts.length, ord = it.opts.slice().sort((a, b) => +a.k - +b.k);
        const em = ord.reduce((s_, o) => s_ + +o.k * o.m, 0), eb = ord.reduce((s_, o) => s_ + +o.k * o.b, 0);
        meter.cols.forEach((c_, j) => {
          const o = ord[j];
          c_.col.style.display = o ? "" : "none";
          c_.col.classList.remove("hit", "gold");
          if (!o) return;
          c_.m.style.height = `${(o.m * 100).toFixed(1)}%`;
          c_.b.style.height = `${(o.b * 100).toFixed(1)}%`;
        });
        meter.tm.style.left = `${(em / (n - 1) * 100).toFixed(1)}%`;
        meter.tb.style.left = `${(eb / (n - 1) * 100).toFixed(1)}%`;
        meter.ans.textContent = "";
      } else {
        if (meter && meter.big) {
          meter.big.className = "st-big";
          meter.big.firstChild.textContent = "…";
          meter.big.lastChild.textContent = "";
        }
        rows.forEach((r, j) => {
          const o = it.opts[j];
          r.row.style.visibility = o ? "visible" : "hidden";
          r.row.classList.remove("hit", "gold");
          if (!o) return;
          r.l.textContent = it.type === "noul" ? o.t : short(elText(o.t), 70);
          r.l.title = o.t;
          r.v.textContent = p2(o.m);
          r.m.style.left = `${(o.m * 100).toFixed(1)}%`;
          r.b.style.left = `${(o.b * 100).toFixed(1)}%`;
        });
      }
      const frameMs = 90, playFor = video ? pr.ims.length * frameMs : 0;
      later(() => {
        dith.style.display = "none";
        img.src = pr.ims[0].src;
        img.style.opacity = "1";
      }, 130);
      if (video) pr.ims.forEach((im, j) => later(() => {
        img.src = im.src;
      }, 130 + j * frameMs));
      const decideAt = 130 + Math.max(300, playFor);
      later(() => {
        const mk = mark(it);
        if (kind === "score") {
          const ord = it.opts.slice().sort((a, b) => +a.k - +b.k), n = ord.length;
          ord.forEach((o, j) => {
            if (o.k === mk.top.k) meter.cols[j].col.classList.add("hit");
            if (o.k === it.gold) meter.cols[j].col.classList.add("gold");
          });
          meter.ans.textContent = mk.top.t;
          verdict.textContent = say(`Top level ${+mk.top.k + 1} of ${n} at ${p2(mk.top.m)}. The truth is level ${+it.gold + 1}, ${LEVEL_OFF[mk.dist] || `${mk.dist} levels off`}.`, `最高是第 ${+mk.top.k + 1} 级（共 ${n} 级），概率 ${p2(mk.top.m)}。真实答案是第 ${+it.gold + 1} 级，${LEVEL_OFF[mk.dist] || `差 ${mk.dist} 级`}。`);
        } else {
          const topRow = rows[0];
          topRow.row.classList.add("hit");
          if (meter && meter.big) {
            const t0 = it.opts[0];
            meter.big.firstChild.textContent = t0.t;
            meter.big.lastChild.textContent = p2(t0.m);
            meter.big.className = "st-big on";
            [0.25, 0.55, 0.8, 1].forEach((f, j) => later(() => {
              meter.big.lastChild.textContent = p2(t0.m * f);
            }, j * 55));
          }
          const g = it.opts.findIndex((o) => o.k === it.gold);
          if (g >= 0) rows[g].row.classList.add("gold");
          const t = it.opts[0];
          verdict.textContent = it.type === "noul" ? say(`${MAIN_D} says ${t.t} at ${p2(t.m)}. Before training, ${BASE} said ${it.base_pick.t} at ${p2(it.base_pick.p)}.`, `${MAIN_D} 答 ${t.t}，概率 ${p2(t.m)}。训练前 ${BASE} 答 ${it.base_pick.t}，概率 ${p2(it.base_pick.p)}。`) : say(`${MAIN_D} picks this at ${p2(t.m)}, one of ${it.n} options. Before training, ${BASE} picked ${short(elText(it.base_pick.t), 60).replace(/\.$/, "")} at ${p2(it.base_pick.p)}.`, `${MAIN_D} 以 ${p2(t.m)} 选了这一项，共 ${it.n} 个选项。训练前 ${BASE} 以 ${p2(it.base_pick.p)} 选了 ${short(elText(it.base_pick.t), 60).replace(/\.$/, "")}。`);
        }
      }, decideAt);
      later(() => {
        const mk = mark(it);
        stamp.textContent = kind === "score" ? mk.ok ? say("✓ exact", "✓ 准确") : mk.near ? say("± one level", "± 差一级") : say(`✗ ${mk.dist} off`, `✗ 差 ${mk.dist} 级`) : mk.ok ? say("✓ right", "✓ 对") : say("✗ wrong", "✗ 错");
        stamp.className = "st-stamp on" + (mk.ok ? "" : mk.near ? " near" : " bad");
        const th = h("div", { class: "st-h" + (mk.ok ? "" : mk.near ? " near" : " bad") }, h("img", { src: pr.ims[Math.floor(pr.ims.length / 2)].src, alt: "" }), h("span", {}, mk.ok ? "✓" : mk.near ? "±" : "✗"));
        hist.prepend(th);
        while (hist.children.length > 14) hist.lastChild.remove();
        decisions++;
        document.dispatchEvent(new CustomEvent("omni-decision", { detail: { kind, ok: mk.ok, sym: mk.ok ? "✓" : mk.near ? "±" : "✗", text: kind === "score" ? say(`level ${+mk.top.k + 1}`, `第 ${+mk.top.k + 1} 级`) : it.opts[0].t, p: kind === "score" ? mk.top.m : it.opts[0].m } }));
      }, decideAt + 280);
      later(() => {
        i = (k + 1) % items.length;
        next();
      }, decideAt + (video ? 1100 : 1150));
    }
    function next() {
      timers.forEach(clearTimeout);
      timers = [];
      if (paused || !(inView || osSeen) || document.hidden) {
        waiting = true;
        return;
      }
      waiting = false;
      show(i);
    }
    prep(0).then(() => {
      waiting = true;
      if (inView || osSeen) next();
    });
    return w;
  }
  function streams(S) {
    const box = $("#streams");
    if (!S.streams || !box) {
      const sec = $("#streams-wrap");
      if (sec) sec.remove();
      return;
    }
    box.replaceChildren(...STREAMS.map(([k, l, sub, noun]) => streamWindow(k, l, sub, noun, S.streams[k] || [], S)).filter(Boolean));
  }
  const andList = (xs) => xs.length < 2 ? xs.join("") : say(`${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`, `${xs.slice(0, -1).join("、")} 和 ${xs[xs.length - 1]}`);
  function news(R) {
    const done = R.models.filter((m) => m.trained && m.status === "done" && !/ v1$/.test(m.name));
    const wait = R.models.filter((m) => m.trained && m.status !== "done");
    const big = R.models.find((m) => m.name === MAIN && m.status === "done");
    const think = R.models.find((m) => /thinking/.test(m.name) && m.test);
    const items = [];
    if (done.length) items.push(say(`${andList(done.map((m) => disp(m.name)))} ${done.length > 1 ? "are" : "is"} trained and scored.`, `${andList(done.map((m) => disp(m.name)))} 已训练并完成评测。`));
    if (wait.length) items.push(say(`${andList(wait.map((m) => disp(m.name)))} ${wait.length > 1 ? "are" : "is"} waiting for ${wait.length > 1 ? "their" : "its"} test scores.`, `${andList(wait.map((m) => disp(m.name)))} 还在等测试分数。`));
    if (big) items.push(say(`${MAIN_D} gets ${pct(big.test.acc)} on ${int(big.test.n)} test questions and ${f1(big.db_hard)} on DecisionBench hard.`, `${MAIN_D} 在 ${int(big.test.n)} 道测试题上得 ${pct(big.test.acc)}，DecisionBench hard 得 ${f1(big.db_hard)}。`));
    if (big && think) items.push(say(`On the same test questions, Qwen3.8-27B in thinking mode gets ${pct(think.test.acc)}, writing about 5,000 tokens per state.`, `同样的测试题，思考模式的 Qwen3.8-27B 得 ${pct(think.test.acc)}，每个状态要写大约 5,000 个 token。`));
    $("#news").replaceChildren(h("ul", {}, items.map((t) => h("li", {}, t))), h("p", { style: "margin-top:14px" }, h("a", { class: "btn light", href: "#results" }, say("Read the results", "看结果"))));
  }
  async function desktop(R, S) {
    const os = $("#os"), canvas = $("#os-field");
    const paint = () => drawField(canvas, 7);
    let rz, lastW = 0, lastH = 0;
    new ResizeObserver(() => {
      if (Math.abs(os.clientWidth - lastW) < 2 && Math.abs(os.clientHeight - lastH) < 2) return;
      lastW = os.clientWidth;
      lastH = os.clientHeight;
      clearTimeout(rz);
      rz = setTimeout(paint, 120);
    }).observe(os);
    const put = (w, left, topPx, width) => {
      w.style.left = left;
      w.style.top = topPx + "px";
      w.style.width = width + "px";
      os.append(w);
      draggable(w, os);
      return w;
    };
    const rows = [];
    const m27 = R.models.find((m) => m.name === MAIN && m.status === "done");
    const b27 = R.models.find((m) => m.name === BASE && m.status === "done");
    const jev = R.external.find((x) => x.name.startsWith("Jev 1.13"));
    const jomni = R.models.find((m) => m.name.startsWith("Jev-Omni") && m.status === "done") || R.external.find((x) => x.name.startsWith("Jev-Omni"));
    const cand = [[m27, MAIN_D, true], [jev, "Jev 1.13", false], [jomni, "Jev-Omni", false], [b27, BASE, false]].filter((c) => c[0]);
    const cols = [["db_medium", "DB medium", f1, "DecisionBench medium", 100], ["db_hard", "DB hard", f1, "DecisionBench hard", 100], ["typesafe_jev354", "TypeSafe", f3, say("TypeSafe (354 rows)", "TypeSafe（354 行）"), 1]];
    const best = {};
    for (const [k] of cols) best[k] = Math.max(...cand.map((c) => c[0][k] ?? -1));
    const heads = cols.map((c) => h("span", { class: "n" }, c[1]));
    rows.push(h("div", { class: "board-row head" }, h("span", {}, say("one forward pass", "一次前向")), heads));
    const cells = [], marks = [];
    for (const [m, name, ours] of cand) {
      const cs = cols.map(([k, , f]) => h("span", { class: "n", style: m[k] === best[k] ? "color:#d45bb6" : "" }, m[k] == null ? "·" : f(m[k])));
      const mk = h("i");
      cells.push(cs);
      marks.push(mk);
      rows.push(h("div", { class: "board-row" }, h("span", { class: "name" }, ours ? h("b", {}, name) : name), cs, h("div", { class: "track" }, mk)));
    }
    const note = h("p", { class: "small", style: "margin-top:8px" });
    let mi = 0;
    const setMetric = (j) => {
      const [k, , , full, top_] = cols[j];
      heads.forEach((e, x) => e.classList.toggle("on", x === j));
      cells.forEach((cs) => cs.forEach((e, x) => e.classList.toggle("on", x === j)));
      cand.forEach(([m], r) => {
        const v = m[k];
        marks[r].style.opacity = v == null ? "0" : "1";
        if (v != null) marks[r].style.left = `${(v / top_ * 100).toFixed(1)}%`;
      });
      note.textContent = say(`Markers show ${full} on a 0 to ${top_} track. Jev 1.13 is text only.`, `标记表示 ${full}，刻度 0 到 ${top_}。Jev 1.13 只支持文本。`);
    };
    setMetric(1);
    put(win("scoreboard 1.0", [h("div", {}, rows), note]), "23%", 60, 520);
    visible(os, (v) => {
      os._seen = v;
      osSeen = v;
      document.dispatchEvent(new Event("omni-os"));
    }, 0.05);
    setInterval(() => {
      if (os._seen && !document.hidden) {
        mi = (mi + 1) % cols.length;
        setMetric(mi);
      }
    }, 1300);
    const mon = S.demos.monitor.find((e) => /phone/i.test(e.state.platform) && e.questions[0].gold === "no") || S.demos.monitor[0];
    const q0 = mon.questions[0];
    const shotBox = h("div", { style: "background:#f9cadd" }, h("p", { class: "px", style: "padding:40px 8px;font-size:15px" }, say("dithering…", "像素化中…")));
    const [yk, yp] = top(q0.probs[MAIN]);
    put(win("screen.pict", [shotBox, h("p", { class: "px", style: "font-size:14px;line-height:1.3;margin-top:6px" }, say(`phone run at step ${mon.state.steps_so_far ?? mon.meta.cut_step}. will it finish? ${yk} ${p2(yp)}`, `手机运行到第 ${mon.state.steps_so_far ?? mon.meta.cut_step} 步。能完成吗？${yk} ${p2(yp)}`))], { cls: "pict" }), "1.5%", 60, 230).querySelector(".win-b").style.padding = "6px";
    Promise.all(mon.media.filter((m) => m.type === "image").map((m) => loadImg(m.src))).then((ims) => {
      const cs = ims.map((im) => [0, 1, 2].map(() => atkinson(im, 112, void 0, 26)));
      let k = 0, f = 0;
      shotBox.replaceChildren(cs[0][0]);
      setInterval(() => {
        if (!os._seen || document.hidden) return;
        f++;
        if (f % 5 === 0) k = (k + 1) % cs.length;
        shotBox.replaceChildren(cs[k][f % 3]);
      }, 170);
    }).catch(() => {
    });
    const sizes = R.models.filter((m) => m.trained && !/ v1$/.test(m.name));
    const sizeBox = h(
      "div",
      { class: "sizes" },
      h("ul", {}, sizes.map((m) => h(
        "li",
        {},
        h("span", {}, m.name.replace(/^OneJev-/, "")),
        m.status === "done" ? blocks(m.test.acc * 100, 100, "mini", "") : h("span", {}, say("soon", "即将")),
        h("span", { class: "v" }, m.status === "done" ? pct(m.test.acc) : "")
      ))),
      h("p", { class: "small", style: "margin-top:8px" }, say("test accuracy, percent", "测试准确率（%）"))
    );
    put(win("sizes", sizeBox), "71%", 60, 300);
    buildOnView(sizeBox);
    const rec = S.demos.recipe[0], rq = rec.questions[0];
    const mov = h("div", { style: "aspect-ratio:16/9;background:#f9cadd" });
    const [rk, rp] = top(rq.probs[MAIN]);
    put(win("egg.mov", [mov, h("p", { class: "px", style: "font-size:14px;line-height:1.3;margin-top:6px" }, say("now: ", "现在：") + `${short(optText(rq, rk), 40)} ${p2(rp)}`)]), "48.5%", 420, 330).querySelector(".win-b").style.padding = "6px";
    Promise.all(rec.media[0].frames.map(loadImg)).then((ims) => {
      const cs = ims.map((im) => atkinson(im, 160));
      let k = 0;
      mov.replaceChildren(cs[0]);
      setInterval(() => {
        k = (k + 1) % cs.length;
        mov.replaceChildren(cs[k]);
      }, 260);
    }).catch(() => {
    });
    const bigEye = h("div", { class: "big-eye", style: "left:77%;top:548px;width:230px" });
    os.append(bigEye);
    eyeSVG(bigEye);
    put(win("onejev.txt", h("p", { class: "px", style: "font-size:15px;line-height:1.45" }, say("OneJev, version 1.0", "OneJev 1.0 版"), h("br"), "Apache-2.0", h("br"), say(`${sizes.length} sizes, one API`, `${sizes.length} 个尺寸，一套 API`))), "1.5%", 692, 230);
    const ICONS = {
      demos: ["############", "#..........#", "#...m......#", "#...mmm....#", "#...mmmmm..#", "#...mmm....#", "#...m......#", "#..........#", "############", "....####...."],
      results: ["............", "........mm..", "........mm..", "....mm..mm..", "....mm..mm..", "mm..mm..mm..", "mm..mm..mm..", "mm..mm..mm..", "############", "............"],
      api: ["############", "#mmmmmmmmmm#", "#..........#", "#.#........#", "#..#.......#", "#.#..###...#", "#..........#", "#..........#", "############", "............"]
    };
    const icon = (key, label, href, left, y) => {
      const g = ICONS[key], svg2 = sv("svg", { viewBox: `0 0 ${g[0].length} ${g.length}`, "shape-rendering": "crispEdges" });
      g.forEach((row, yy) => [...row].forEach((ch, xx) => {
        if (ch !== ".") svg2.append(sv("rect", { x: xx, y: yy, width: 1, height: 1, fill: ch === "m" ? "#d45bb6" : "#1e1e1e" }));
      }));
      const a = h("a", { class: "ico", href, style: `left:${left};top:${y}px;text-decoration:none` }, svg2, h("span", {}, label));
      os.append(a);
    };
    icon("demos", say("demos", "演示"), "#demos", "24%", 640);
    icon("results", say("results", "结果"), "#results", "31%", 640);
    icon("api", "api", "#start", "38%", 640);
    const dateEl = h("span"), timeEl = h("span", { style: 'font-size:28px;font-family:"Press Start 2P",monospace;margin-top:6px' });
    const tickClock = () => {
      const d = new Date();
      dateEl.textContent = d.toLocaleDateString(say("en-US", "zh-CN"), { weekday: "long", month: "short", day: "numeric", year: "numeric" });
      timeEl.textContent = d.toLocaleTimeString("en-GB");
    };
    tickClock();
    setInterval(tickClock, 1e3);
    put(win("clock 1.0", h("p", { class: "px", style: "font-size:16px;line-height:1.3;display:grid" }, dateEl, timeEl)), "71%", 300, 300);
    const nEl = h("span", { style: 'font-size:30px;line-height:1.2;font-family:"Press Start 2P",monospace;color:#d45bb6' }, "0"), lastEl = h("span", {}, say("waiting for the streams", "等待实时流"));
    put(win("decisions 1.0", h("p", { class: "px", style: "font-size:15px;line-height:1.4;display:grid;gap:6px" }, nEl, lastEl)), "23%", 420, 270);
    document.addEventListener("omni-decision", (e) => {
      nEl.textContent = int(decisions);
      const d = e.detail;
      lastEl.textContent = `${d.kind}: ${short(String(d.text), 22)} ${p2(d.p)} ${d.sym}`;
      nEl.classList.remove("bump");
      void nEl.offsetWidth;
      nEl.classList.add("bump");
    });
  }
  function statements(R, S) {
    const sp = S.speed;
    if (sp) {
      const words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"];
      const zhWords = ["零", "一", "两", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"];
      $("#speed-title").textContent = say(`${words[sp.screenshot.q] || sp.screenshot.q} Questions, One Look, ${sp.screenshot.batch} ms.`, `看一眼，答${zhWords[sp.screenshot.q] || sp.screenshot.q}题，${sp.screenshot.batch} 毫秒`);
      $("#speed-note").append(
        say(`*${sp.model} on one ${sp.gpu}, a state with one screenshot (${int(sp.screenshot.tokens)} tokens), median of five warm runs with media decoding included. `, `*${sp.model}，一张 ${sp.gpu}，状态里有一张截图（${int(sp.screenshot.tokens)} 个 token），预热后跑五次取中位数，包含媒体解码。`),
        h("a", { href: "#race-desk" }, say("(Race)", "（看对比）"))
      );
    }
    if (S.mix && S.mix.total) {
      const g = S.mix.groups, seen = ["gui", "image", "video_short", "video_long"].reduce((s, k) => s + (g[k]?.rows || 0), 0);
      $("#col-looks").textContent = say(`A state can hold screenshots, photos and video frames. ${Math.round(100 * seen / S.mix.total)} percent of the ${int(S.mix.total)} training questions are about something the model has to look at.`, `状态里可以放截图、照片和视频帧。${int(S.mix.total)} 道训练题里有 ${Math.round(100 * seen / S.mix.total)}% 需要模型去看图或看视频。`);
      const tt = $("#train-text").firstChild;
      tt.textContent = ZH ? tt.textContent.replace("同一批问题", `同一批 ${int(S.mix.total)} 道问题`) : tt.textContent.replace("all trained on the same questions", `all trained on the same ${int(S.mix.total)} questions`);
    }
  }
  function race(S) {
    const sp = S.speed;
    if (!sp) {
      $("#race-desk").remove();
      return;
    }
    const box = $("#race"), foot = $("#race-foot");
    let timers = [];
    const qs = sp.questions;
    function run(kind) {
      timers.forEach(clearTimeout);
      timers.forEach(cancelAnimationFrame);
      timers = [];
      const d = sp[kind], n = qs.length;
      const mk = (title, head) => {
        const rows = qs.map((q) => h("div", { class: "qrow" }, h("span", { class: "box" }), h("span", {}, q)));
        const clock = h("span", { class: "k" }, "0.000 s");
        const t = h("div", { class: "term" }, h("div", { class: "dim" }, head), h("div", { style: "margin:8px 0" }, rows), h("div", {}, say("elapsed ", "耗时 "), clock));
        return { el: win(title, t, { bcls: "flush" }), rows, clock };
      };
      const media = kind === "screenshot" ? say("the screenshot", "截图") : say("the video", "视频");
      const A = mk(say("OneJev, one request", "OneJev，一次请求"), say(`# ${media} and all ${n} questions in one request`, `# ${media}和全部 ${n} 个问题放在一次请求里`));
      const B = mk(say("OneJev, ten requests", "OneJev，十次请求"), say(`# ${media} and one question, sent ${n} times`, `# ${media}加一个问题，发 ${n} 次`));
      box.replaceChildren(A.el, B.el);
      foot.replaceChildren(
        h("div", { class: "stat" }, h("b", {}, say("one request", "一次请求")), h("span", {}, say(`${n} questions in ${(d.batch / 1e3).toFixed(3)} s`, `${n} 个问题 ${(d.batch / 1e3).toFixed(3)} 秒`))),
        h("div", { class: "stat" }, h("b", {}, say("ten requests", "十次请求")), h("span", {}, say(`${n} questions in ${(d.separate / 1e3).toFixed(3)} s`, `${n} 个问题 ${(d.separate / 1e3).toFixed(3)} 秒`))),
        h("div", { class: "stat" }, h("b", {}, say("one question alone", "只问一个问题")), h("span", {}, say(`${(d.one / 1e3).toFixed(3)} s`, `${(d.one / 1e3).toFixed(3)} 秒`))),
        h("button", { class: "btn", type: "button", style: "margin-left:auto", onclick: () => run(kind) }, say("run again", "再跑一次"))
      );
      const t0 = performance.now();
      const tick = () => {
        const t = performance.now() - t0;
        A.clock.textContent = (Math.min(t, d.batch) / 1e3).toFixed(3) + " s";
        B.clock.textContent = (Math.min(t, d.separate) / 1e3).toFixed(3) + " s";
        if (t >= d.batch) A.rows.forEach((r) => r.firstChild.classList.add("on"));
        B.rows.forEach((r, i) => {
          if (t >= d.separate * (i + 1) / n) r.firstChild.classList.add("on");
        });
        if (t < d.separate) timers.push(requestAnimationFrame(tick));
        else timers.push(setTimeout(() => {
          if (inView) run(kind);
          else again = kind;
        }, 2600));
      };
      timers.push(requestAnimationFrame(tick));
    }
    let started = false, current = "screenshot", inView = false, again = null;
    visible($("#race-desk"), (v) => {
      inView = v;
      if (v && !started) {
        started = true;
        run(current);
      } else if (v && again) {
        const k = again;
        again = null;
        run(k);
      }
    }, 0.3);
    tabs($("#race-tabs"), [say("screenshot", "截图"), say("video", "视频")], (i) => {
      current = ["screenshot", "video"][i];
      if (started) run(current);
      else box.replaceChildren();
    });
  }
  function thoughtOf(s) {
    let t = s.thought || "";
    const m = /Thought:\s*([\s\S]*?)(?:'''|$)/.exec(s.action || "");
    if (!t && m) t = m[1];
    t = t.replace(/<thinking>[\s\S]*?<\/thinking>/g, " ").replace(/<tool_call>[\s\S]*/g, " ").replace(/<\/?[a-z_]+>/g, " ");
    t = t.replace(/\s+/g, " ").trim();
    if (!t) {
      const inner = /<thinking>([\s\S]*?)<\/thinking>/.exec(s.thought || "");
      if (inner) t = inner[1].replace(/\s+/g, " ").trim();
    }
    return t;
  }
  function actionOf(s) {
    const a = s.action || "";
    const code = a.split("'''").pop().replace(/import \w+\s*/g, "").replace(/\s+/g, " ").trim();
    return code || a;
  }
  function truthOutcome(q, then) {
    const yes = q.gold === "yes";
    const bits = [h("b", {}, yes ? say("The run finished the task.", "这次运行完成了任务。") : say("The run did not finish the task.", "这次运行没有完成任务。")), say(" The benchmark’s own program checked the final state.", "结果由基准自带的程序检查最终状态得出。")];
    if (then) {
      const tq = then.questions[0], [k, p] = top(tq.probs[MAIN]);
      bits.push(say(` It lasted ${then.state.total_steps ?? then.meta.run_steps} steps. Asked about the whole run afterwards, ${MAIN_D} answered ${k} ${p2(p)}.`, `整次运行共 ${then.state.total_steps ?? then.meta.run_steps} 步。事后针对整次运行提问，${MAIN_D} 答 ${k} ${p2(p)}。`));
    }
    return h("div", { class: "truth" }, bits);
  }
  function monitor(S) {
    const ex = S.demos.monitor, out = $("#monitor");
    const count = { phone: 0, desktop: 0 };
    const labels = ex.map((e) => {
      const k = /phone/i.test(e.state.platform) ? "phone" : "desktop";
      count[k]++;
      return `${k === "phone" ? say("phone", "手机") : say("desktop", "桌面")} ${count[k]}`;
    });
    tabs($("#monitor-tabs"), labels, (i) => {
      const e = ex[i], q = e.questions[0], imgs = e.media.filter((m) => m.type === "image");
      const tall = imgs[0].h > imgs[0].w;
      const big = h("img", { src: imgs[imgs.length - 1].src, alt: say("agent screenshot", "agent 截图") });
      const thumbs = h("div", { class: "thumbs" });
      imgs.forEach((m, j) => thumbs.append(h("button", { type: "button", "aria-pressed": j === imgs.length - 1 ? "true" : "false", "aria-label": say(`screen ${j + 1}`, `屏幕 ${j + 1}`), onclick: (ev) => {
        big.src = m.src;
        thumbs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
        ev.currentTarget.setAttribute("aria-pressed", "true");
      } }, h("img", { src: m.src, alt: "" }))));
      const upto = e.state.steps_so_far ?? e.state.total_steps;
      const left = win(say(`${labels[i]}, screens up to step ${upto}`, `${labels[i]}，截至第 ${upto} 步的屏幕`), [h("div", { class: "shot" + (tall ? " tall" : "") }, big), thumbs], { bcls: "flush" });
      const st = e.state;
      const steps = h("ol", { class: "steps" }, (st.steps || []).map((s) => {
        const th = thoughtOf(s);
        return h("li", {}, h("b", {}, s.step), h("span", {}, th ? h("span", { class: "th" }, short(th, 170)) : null, h("span", { class: "ac" }, short(actionOf(s), 110))));
      }));
      const run = win(say("run so far", "到目前为止的运行"), [
        h("p", { class: "small" }, h("b", {}, say("Task. ", "任务：")), st.task),
        e.note ? h("p", { class: "small", style: "margin-top:6px" }, h("b", {}, say("In English. ", "英文译文：")), e.note) : null,
        h("p", { class: "small", style: "margin-top:6px" }, say(`${st.platform}. ${st.omitted_earlier_steps ? `${st.omitted_earlier_steps} earlier steps are summarised and the ` : "The "}last ${st.steps.length} steps are shown.`, `${st.platform}。${st.omitted_earlier_steps ? `前面 ${st.omitted_earlier_steps} 步做了摘要，` : ""}展示最后 ${st.steps.length} 步。`)),
        steps
      ]);
      const pre = h("pre", { class: "term" });
      const term = win("systemone", pre, { bcls: "flush" });
      const ans = win(say("answers", "回答"), [
        h("p", { class: "qline" }, h("span", { class: "px" }, q.question.type), q.question.text),
        answerBars(q),
        truthOutcome(q, e.then)
      ]);
      out.replaceChildren(h("div", { class: "grid-monitor" }, left, h("div", { class: "side" }, run, ans, term)));
      grow(out);
      typeInto(pre, requestText(e, MAIN_D), 6);
    }, 9e3);
  }
  function click(S) {
    const ex = S.demos.click, out = $("#click");
    tabs($("#click-tabs"), ex.map((e) => e.state.website || "page"), (i) => {
      const e = ex[i], q = e.questions[0], m = e.media[0];
      const at = (k) => {
        const r = /center at about \((\d+), (\d+)\)/.exec(optText(q, k));
        return r ? [+r[1] / m.w * 100, +r[2] / m.h * 100] : null;
      };
      const marks = [];
      const add = (k, cls, label) => {
        const p = at(k);
        if (!p) return;
        const right = p[0] > 70;
        marks.push(h("div", { class: `mark ${cls}`, style: `left:${p[0]}%;top:${p[1]}%` }, h("i"), h("span", { style: right ? "left:auto;right:20px" : "" }, label)));
      };
      const [mk, mp] = top(q.probs[MAIN]);
      const [bk, bp] = q.probs[BASE] ? top(q.probs[BASE]) : [null, 0];
      if (bk && bk !== mk) add(bk, "base", `Qwen3.8-27B ${p2(bp)}`);
      add(mk, "main", `${MAIN_D} ${p2(mp)}`);
      if (q.gold !== mk) add(q.gold, "main", say("recorded click", "录下的点击"));
      const shot = h("div", { class: "shot" }, h("img", { src: m.src, alt: say("web page screenshot", "网页截图") }), marks);
      const st = e.state;
      const opts = q.question.options.slice().sort((a, b) => (q.probs[MAIN][b.key] ?? 0) - (q.probs[MAIN][a.key] ?? 0)).slice(0, 6);
      const list = h("ul", { class: "opts" }, opts.map((o) => h(
        "li",
        { class: o.key === q.gold ? "gold" : "" },
        h("div", { class: "row" }, h("span", {}, short(elText(o.text), 80)), h("span", {}, p2(q.probs[MAIN][o.key] ?? 0))),
        h("div", { class: "bar" }, h("i", { style: `width:${((q.probs[MAIN][o.key] ?? 0) * 100).toFixed(1)}%` })),
        q.probs[BASE] ? h("div", { class: "bar dither", style: "height:8px;margin-top:2px" }, h("i", { style: `width:${((q.probs[BASE][o.key] ?? 0) * 100).toFixed(1)}%` })) : null
      )));
      const side = h(
        "div",
        { class: "side", style: "display:grid;gap:14px" },
        win(say("task", "任务"), [
          h("p", { class: "small" }, h("b", {}, say(`${st.website}. `, `${st.website}：`)), st.task),
          h("ol", { class: "steps" }, (st.previous_actions || []).map((a, j) => h("li", {}, h("b", {}, j + 1), h("span", {}, a))))
        ]),
        win(say(`${q.question.n_options} elements`, `${q.question.n_options} 个元素`), [h("p", { class: "small", style: "margin-bottom:10px" }, say(`The six most likely for ${MAIN_D}. Magenta blocks are ${MAIN_D}, grey blocks are Qwen3.8-27B before training, and ■ marks the element the recorded demonstration clicked.`, `${MAIN_D} 认为最可能的六个。品红色块是 ${MAIN_D}，灰色块是训练前的 Qwen3.8-27B，■ 标出录下的演示实际点击的元素。`)), list])
      );
      const pts = [mk, bk, q.gold].filter(Boolean).map(at).filter(Boolean).map((p) => [p[0] / 100, p[1] / 100]);
      const ar = m.h / m.w * 16 / 9;
      const sx = Math.max(...pts.map((p) => p[0])) - Math.min(...pts.map((p) => p[0])) + 0.16;
      const sy = (Math.max(...pts.map((p) => p[1])) - Math.min(...pts.map((p) => p[1])) + 0.14) * ar;
      const zoom = Math.max(1.5, Math.min(4, 1 / Math.max(sx, sy)));
      let cx = pts.reduce((a, p) => a + p[0], 0) / pts.length, cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
      cx = Math.min(1 - 0.5 / zoom, Math.max(0.5 / zoom, cx));
      cy = Math.min(1 - 0.5 / (zoom * ar), Math.max(0.5 / (zoom * ar), cy));
      const zw = h(
        "div",
        { class: "zw", style: `width:${zoom * 100}%;left:${50 - cx * zoom * 100}%;top:${50 - cy * zoom * ar * 100}%` },
        h("img", { src: m.src, alt: "" }),
        marks.map((x) => x.cloneNode(true))
      );
      const loupe = win(say(`fatbits, ${zoom.toFixed(1)}x`, `放大镜，${zoom.toFixed(1)}x`), h("div", { class: "loupe" }, zw), { bcls: "flush" });
      out.replaceChildren(h("div", { class: "grid-click" }, h("div", { style: "display:grid;gap:14px" }, loupe, win(say(`${st.website}, next click`, `${st.website}，下一次点击`), shot, { bcls: "flush" })), side));
      grow(out);
    }, 8e3);
  }
  const players = {};
  function stopPlayers(k) {
    (players[k] || []).forEach((p) => p.stop());
    players[k] = [];
  }
  function compareLine(q, models = [[BASE, say("Qwen3.8-27B before training", "Qwen3.8-27B（训练前）")], [SMALL, SMALL_D]]) {
    return models.filter(([n]) => q.probs[n]).map(([n, l]) => {
      const [k, p] = top(q.probs[n]), ok = k === q.gold;
      return h("p", { class: "small", style: "margin-top:6px" }, h("b", {}, say(`${l}. `, `${l}：`)), say(`${short(optText(q, k), 70)} ${p2(p)}, `, `${short(optText(q, k), 70)} ${p2(p)}，`), ok ? say("right.", "对。") : say("wrong.", "错。"));
    });
  }
  function recipe(S) {
    const ex = S.demos.recipe, out = $("#recipe");
    tabs($("#recipe-tabs"), ex.map((e) => say(`${(e.state.recipe || e.state.task).replace(/\.$/, "")}, ${e.state.elapsed}`, `${(e.state.recipe || e.state.task).replace(/\.$/, "")}，${dur(e.state.elapsed)}`)), (i) => {
      stopPlayers("recipe");
      const e = ex[i];
      const vids = e.media.filter((m) => m.type === "video");
      const holder = h("div");
      const mount = (k) => {
        stopPlayers("recipe");
        const p = player(vids[k].frames, { fps: k === 0 ? 5 : 3 });
        players.recipe.push(p);
        holder.replaceChildren(p.el);
      };
      const switcher = vids.length > 1 ? h("div", { class: "pager" }, [say("whole recording", "完整录像"), say("last 8 seconds", "最后 8 秒")].map((l, k) => h("button", { type: "button", "aria-pressed": k === 0 ? "true" : "false", onclick: (ev) => {
        ev.currentTarget.parentNode.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
        ev.currentTarget.setAttribute("aria-pressed", "true");
        mount(k);
      } }, l))) : null;
      const left = h(
        "div",
        { style: "display:grid;gap:14px" },
        win(say(`${e.state.recipe || e.state.task.replace(/\.$/, "")}, ${e.state.elapsed} in`, `${e.state.recipe || e.state.task.replace(/\.$/, "")}，第 ${dur(e.state.elapsed)}`), holder, { bcls: "flush", bar: switcher }),
        e.state.recipe_steps ? win(say("recipe card", "菜谱卡片"), h("ol", { class: "steps", style: "max-height:none;font-size:12.5px" }, e.state.recipe_steps.map((s) => {
          const m = /^(\d+)\.\s*(.*)$/.exec(s);
          return h("li", {}, h("b", {}, m ? m[1] : ""), h("span", {}, m ? m[2] : s));
        }))) : null
      );
      mount(0);
      const rpre = h("pre", { class: "term" });
      left.append(win("systemone", rpre, { bcls: "flush" }));
      typeInto(rpre, requestText(e, MAIN_D), 6);
      const right = h("div", { style: "display:grid;gap:14px" });
      for (const q of e.questions) {
        const P = q.probs[MAIN];
        if (q.task === "current_step" || q.task === "next_step") {
          const opts = q.question.options;
          const list = h("ol", { class: "steplist" }, opts.map((o) => h("li", { class: o.key === q.gold ? "gold" : "" }, h("span", {}, o.text), h("span", { class: "b" }, h("i", { style: `width:${((P[o.key] ?? 0) * 100).toFixed(1)}%` })))));
          const shown = opts.length < q.question.n_options ? say(` The ${opts.length} most likely of ${q.question.n_options} steps are shown.`, `只显示 ${q.question.n_options} 步里最可能的 ${opts.length} 步。`) : "";
          right.append(win(q.task === "current_step" ? say("now", "现在") : say("next", "下一步"), [h("p", { class: "qline" }, h("span", { class: "px" }, "choice"), q.question.text), h("p", { class: "small", style: "margin-bottom:8px" }, say(`Bars are ${MAIN_D}.${shown}`, `柱条是 ${MAIN_D}。${shown}`)), list, compareLine(q)]));
        } else if (q.task === "progress_level") {
          const opts = q.question.options.slice().sort((a, b) => +a.key - +b.key);
          const g = h("div", { class: "gauge" }, opts.map((o) => h("div", { class: o.key === q.gold ? "g" : "" }, h("i", { style: `height:${((P[o.key] ?? 0) * 100).toFixed(1)}%` }))));
          const lab = h("div", { class: "gauge-l" }, say(["under 1/5", "1/5", "2/5", "3/5", "4/5 +"], ["不到 1/5", "1/5", "2/5", "3/5", "4/5 以上"]).slice(0, opts.length).map((t) => h("span", {}, t)));
          right.append(win(say("progress", "进度"), [
            h("p", { class: "qline" }, h("span", { class: "px" }, "score"), q.question.text),
            g,
            lab,
            h("p", { class: "small", style: "margin-top:8px" }, h("b", {}, say("Truth. ", "真实答案：")), optText(q, q.gold), say(" It has the pink outline; bar heights are ", " 粉色描边的那一格就是它，柱高是 "), MAIN_D, say(".", " 给的概率。")),
            compareLine(q)
          ]));
        }
      }
      out.replaceChildren(h("div", { class: "grid-recipe" }, left, right));
      grow(out);
    }, 1e4);
  }
  const CLIP_NAMES = ZH ? { ssv2: "手部动作", diving48: "跳水", msrvtt: "描述", clevrer: "碰撞", konvid1k: "视频质量", nextqa: "视频问答" } : { ssv2: "hand actions", diving48: "dives", msrvtt: "captions", clevrer: "collisions", konvid1k: "video quality", nextqa: "video QA" };
  function clips(S) {
    const ex = S.demos.clips, out = $("#clip");
    const seen = {};
    const labels = ex.map((e) => {
      const b = CLIP_NAMES[e.source] || e.source;
      seen[b] = (seen[b] || 0) + 1;
      return seen[b] > 1 ? `${b} ${seen[b]}` : b;
    });
    tabs($("#clip-tabs"), labels, (i) => {
      stopPlayers("clip");
      const e = ex[i], q = e.questions[0];
      const p = player(e.media[0].frames, { fps: 6 });
      players.clip.push(p);
      const [k, pr] = top(q.probs[MAIN]);
      const ans = q.question.type === "noul" ? k === "yes" ? say("Yes.", "是。") : say("No.", "否。") : optText(q, k);
      const right = win(say("answer", "回答"), [
        h("p", { class: "qline" }, h("span", { class: "px" }, q.question.type), q.question.text),
        h("p", { class: "big-answer" }, ans),
        h("p", { class: "small", style: "margin-bottom:12px" }, say(`${MAIN_D} gives it ${p2(pr)}, ${nOptions(q)}.`, `${MAIN_D} 给它 ${p2(pr)}，${nOptions(q)}。`)),
        answerBars(q),
        h("div", { class: "truth" }, h("b", {}, say("Truth. ", "真实答案：")), optText(q, q.gold).replace(/\.$/, ""), say(". The label comes with the dataset.", "。标签来自数据集本身。"))
      ]);
      const cpre = h("pre", { class: "term" });
      out.replaceChildren(h("div", { class: "grid-clip" }, h("div", { style: "display:grid;gap:14px" }, win(say(`${labels[i]}, ${e.media[0].frames.length} frames`, `${labels[i]}，${e.media[0].frames.length} 帧`), p.el, { bcls: "flush" }), win("systemone", cpre, { bcls: "flush" })), right));
      grow(out);
      typeInto(cpre, requestText(e, MAIN_D), 6);
    }, 8e3);
  }
  const PHOTO_NAMES = ZH ? { flowers102: "花", stanford_cars: "汽车", cub200: "鸟", fgvc_aircraft: "飞机", oxford_pets: "宠物", gtsrb: "交通标志", eurosat: "卫星图", cauldron_mapqa: "地图", cauldron_figureqa: "图表" } : { flowers102: "flowers", stanford_cars: "cars", cub200: "birds", fgvc_aircraft: "aircraft", oxford_pets: "pets", gtsrb: "traffic signs", eurosat: "satellite", cauldron_mapqa: "maps", cauldron_figureqa: "charts" };
  function photos(S) {
    const box = $("#photos");
    const cards = S.demos.photos.map((e) => {
      const q = e.questions[0], [k2, p] = top(q.probs[MAIN]);
      const base = q.probs[BASE] ? top(q.probs[BASE]) : null;
      const yn = (k) => k === "yes" ? say("Yes", "是") : say("No", "否");
      const ans = q.question.type === "noul" ? yn(k2) : optText(q, k2);
      const bt = base ? q.question.type === "noul" ? yn(base[0]) : optText(q, base[0]) : "";
      return win(say(`${PHOTO_NAMES[e.source] || e.source}, ${nOptions(q)}`, `${PHOTO_NAMES[e.source] || e.source}，${nOptions(q)}`), [
        h("div", { class: "shot" }, h("img", { src: e.media[0].src, alt: "", loading: "lazy" })),
        h(
          "div",
          { style: "padding:10px 11px" },
          h("p", { class: "q" }, q.question.text),
          h("p", { class: "a" }, `${ans} `, h("span", { class: "mono", style: "font-size:13px;font-weight:400" }, p2(p))),
          h("div", { class: "bar", style: "margin-top:5px" }, h("i", { style: `width:${(p * 100).toFixed(1)}%` })),
          base ? h("p", { class: "b2" }, say("Qwen3.8-27B before training said ", "训练前的 Qwen3.8-27B 答 "), base[0] === q.gold ? bt : h("s", {}, bt), say(` ${p2(base[1])}.`, ` ${p2(base[1])}。`)) : null,
          k2 !== q.gold ? h("p", { class: "b2" }, h("b", {}, say("Truth. ", "真实答案：")), optText(q, q.gold)) : null
        )
      ], { bcls: "flush" });
    });
    box.replaceChildren(...cards);
    let k = 0, seen = false;
    visible(box, (v) => {
      seen = v;
    }, 0.1);
    setInterval(() => {
      if (!seen || document.hidden) return;
      const c = cards[k++ % cards.length], im = c.querySelector(".shot img"), shot = c.querySelector(".shot");
      if (!im || !im.complete) return;
      const dz = atkinson(im, 150);
      dz.className = "flash-dither";
      shot.append(dz);
      c.classList.add("look");
      setTimeout(() => {
        dz.remove();
        c.classList.remove("look");
        grow(c);
      }, 170);
    }, 2600);
  }
  function svg(w, hgt, inner) {
    return h("div", { html: `<svg viewBox="0 0 ${w} ${hgt}" role="img" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">${inner}</svg>` });
  }
  function calibration(R, S) {
    const pairs = [[MAIN, BASE], [SMALL, "Qwen3.5-4B"]].filter(([a, b]) => S.calibration[a] && S.calibration[b]);
    if (!pairs.length) return;
    const ece = (n) => R.models.find((m) => m.name === n)?.test?.ece;
    const [a0, b0] = pairs[0];
    if (ece(a0) != null && ece(b0) != null) $("#cal-title").textContent = say(`Calibration Error Drops From ${f3(ece(b0))} To ${f3(ece(a0))}`, `校准误差从 ${f3(ece(b0))} 降到 ${f3(ece(a0))}`);
    function draw(i) {
      const [a, b] = pairs[i], W = 560, H = 440, L = 58, T = 18, PW = W - L - 18, PH = H - T - 58;
      const X = (v) => L + v * PW, Y = (v) => T + (1 - v) * PH;
      let s = `<rect x="${L}" y="${T}" width="${PW}" height="${PH}" fill="#fefefe" stroke="#1e1e1e" stroke-width="2"/>`;
      for (let k = 1; k < 10; k++) for (let j = 1; j < 10; j++) s += `<rect x="${X(k / 10) - 1}" y="${Y(j / 10) - 1}" width="2" height="2" fill="#dcdcdc"/>`;
      for (let k = 0; k < 40; k++) {
        const v = (k + 0.5) / 40;
        s += `<rect x="${X(v) - 1.5}" y="${Y(v) - 1.5}" width="3" height="3" fill="#c4c4c4"/>`;
      }
      for (const [n, ours] of [[b, false], [a, true]]) {
        const pts = S.calibration[n].bins.filter((x) => x.n >= 20);
        s += `<polyline fill="none" stroke="${ours ? "#d45bb6" : "#6b6b6b"}" stroke-width="${ours ? 3 : 2}" ${ours ? "" : 'stroke-dasharray="6 4"'} points="${pts.map((p) => `${X(p.conf)},${Y(p.acc)}`).join(" ")}"/>`;
        for (const p of pts) s += ours ? `<rect x="${X(p.conf) - 7}" y="${Y(p.acc) - 7}" width="14" height="14" fill="#d45bb6" stroke="#1e1e1e" stroke-width="2"/>` : `<rect x="${X(p.conf) - 5}" y="${Y(p.acc) - 5}" width="10" height="10" fill="#fefefe" stroke="#6b6b6b" stroke-width="2"/>`;
      }
      for (let k = 0; k <= 10; k += 2) s += `<text x="${X(k / 10)}" y="${T + PH + 22}" text-anchor="middle">${(k / 10).toFixed(1)}</text><text x="${L - 8}" y="${Y(k / 10) + 6}" text-anchor="end">${(k / 10).toFixed(1)}</text>`;
      s += `<text x="${L + PW / 2}" y="${H - 8}" text-anchor="middle">${say("confidence of the top option", "最高选项的置信度")}</text><text transform="translate(16 ${T + PH / 2}) rotate(-90)" text-anchor="middle">${say("share right", "答对比例")}</text>`;
      s += `<rect x="${L + 14}" y="${T + 14}" width="12" height="12" fill="#d45bb6" stroke="#1e1e1e" stroke-width="2"/><text x="${L + 34}" y="${T + 25}">${disp(a)}</text><rect x="${L + 14}" y="${T + 36}" width="10" height="10" fill="#fefefe" stroke="#6b6b6b" stroke-width="2"/><text x="${L + 34}" y="${T + 47}">${b}${say(", before training", "，训练前")}</text>`;
      $("#cal-chart").replaceChildren(svg(W, H, s));
      const hi = (n) => {
        const bs2 = S.calibration[n].bins.slice(9);
        const c = bs2.reduce((x, y) => x + y.n, 0);
        return { n: c, acc: bs2[0].acc, conf: bs2[0].conf, share: c / S.calibration[n].n };
      };
      const A = hi(a), B = hi(b);
      $("#cal-text").replaceChildren(
        h("p", { class: "lead" }, say(`Every test answer is placed by the probability the model gave its top option. Each square shows how often answers at that confidence were right; on the grey diagonal the two agree.`, `每个测试答案按模型给最高选项的概率归位。每个方块表示这个置信度下答对的比例；落在灰色对角线上说明两者一致。`)),
        h("p", { class: "lead" }, say(`${disp(a)} put 0.9 or more on ${int(A.n)} of ${int(S.calibration[a].n)} answers, at ${f3(A.conf)} on average, and ${pct(A.acc)} percent of them were right. ${b} before training was that sure on ${int(B.n)} answers and right on ${pct(B.acc)} percent.`, `${int(S.calibration[a].n)} 个答案里，${disp(a)} 有 ${int(A.n)} 个给了 0.9 以上，平均 ${f3(A.conf)}，其中 ${pct(A.acc)}% 答对。训练前的 ${b} 有 ${int(B.n)} 个答案这么有把握，答对 ${pct(B.acc)}%。`)),
        ece(a) != null ? h("p", { class: "lead" }, say(`Expected calibration error over the ${int(S.calibration[a].n)} questions is ${f3(ece(a))} for ${disp(a)} and ${f3(ece(b))} before training.`, `在这 ${int(S.calibration[a].n)} 道题上，${disp(a)} 的期望校准误差是 ${f3(ece(a))}，训练前是 ${f3(ece(b))}。`)) : null
      );
    }
    const pg = $("#cal-pager");
    pg.replaceChildren();
    const bs = pairs.map((p, i) => h("button", { type: "button", "aria-pressed": i === 0 ? "true" : "false", onclick: () => {
      bs.forEach((x, j) => x.setAttribute("aria-pressed", i === j ? "true" : "false"));
      draw(i);
    } }, p[0].replace("OneJev-", "")));
    pg.append(...bs);
    draw(0);
  }
  function results(R) {
    const models = R.models.filter((m) => !/ v1$/.test(m.name));
    const done = models.filter((m) => m.status === "done");
    const byName = (n) => models.find((m) => m.name === n && m.status === "done");
    const ours = ["27B", "9B", "4B", "0.8B"].map((sz) => models.find((m) => m.trained && m.size === sz && m.status === "done")).filter(Boolean);
    const jev = R.external.find((x) => x.name.startsWith("Jev 1.13"));
    const jomni = byName("Jev-Omni (Gemma 4 12B)") || R.external.find((x) => x.name.startsWith("Jev-Omni"));
    const think = models.find((m) => /thinking/.test(m.name) && m.status === "done");
    const m27 = byName(MAIN);
    if (m27 && jev && m27.db_hard != null) {
      $("#res-title").textContent = say(`${f1(m27.db_hard)} On DecisionBench Hard. Jev 1.13 Scores ${f1(jev.db_hard)}.`, `DecisionBench Hard 得 ${f1(m27.db_hard)}，Jev 1.13 是 ${f1(jev.db_hard)}`);
      $("#res-note").textContent = say(`${MAIN_D}, one forward pass per question, state-macro as on the benchmark card. On TypeSafe's own public rows it gets ${pct(m27.typesafe_jev354)} against Jev's published ${pct(jev.typesafe_jev354)}.`, `${MAIN_D}，每个问题一次前向，按基准卡片的 state-macro 计分。在 TypeSafe 自己公开的题目上，它得 ${pct(m27.typesafe_jev354)}，Jev 公布的成绩是 ${pct(jev.typesafe_jev354)}。`);
    }
    const nTotal = /([\d,]+) rows/.exec(R.test_set || "");
    const any = done.find((m) => m.trained) || done[0];
    let tt = say(`${nTotal ? nTotal[1] : ""} questions from the same sources as training, item-disjoint from it: no trajectory, video or image appears on both sides. ${any ? int(any.test.n) : ""} fit the length limit and are scored. Accuracy of the top option, in percent.`, `${nTotal ? nTotal[1] : ""} 道题，和训练集来源相同，但条目不重叠：没有任何轨迹、视频或图片同时出现在两边。其中 ${any ? int(any.test.n) : ""} 道在长度限制内，参与计分。指标是最高选项的准确率（%）。`);
    if (m27 && think) tt += say(` ${MAIN_D} gets ${pct(m27.test.acc)} on them in one forward pass; Qwen3.8-27B in thinking mode, writing about 5,000 tokens per state, gets ${pct(think.test.acc)}.`, `${MAIN_D} 一次前向得 ${pct(m27.test.acc)}；思考模式的 Qwen3.8-27B 每个状态写大约 5,000 个 token，得 ${pct(think.test.acc)}。`);
    $("#test-text").textContent = tt;
    const cols = [
      ...ours.map((m) => ({ label: disp(m.name), head: ["OneJev", m.size], m, ours: true })),
      jev ? { label: "Jev 1.13", head: ["Jev", "1.13"], m: jev, textOnly: true } : null,
      jomni ? { label: "Jev-Omni 12B", head: ["Jev-Omni", "12B"], m: jomni } : null,
      think ? { label: say("Qwen 27B thinking", "Qwen 27B 思考"), head: ["Qwen 27B", say("thinking", "思考")], m: think } : null
    ].filter(Boolean);
    const x100 = (v) => v == null ? null : v * 100;
    const metrics = {
      test: { title: say("OneJev test set", "OneJev 测试集"), get: (m) => x100(m.test?.acc), imageOnly: true },
      gui: { title: say("GUI agents", "GUI agent"), get: (m) => x100(m.groups?.gui?.acc), imageOnly: true },
      image: { title: say("Images", "图片"), get: (m) => x100(m.groups?.image?.acc), imageOnly: true },
      video_long: { title: say("Long video", "长视频"), get: (m) => x100(m.groups?.video_long?.acc), imageOnly: true },
      video_short: { title: say("Short video", "短视频"), get: (m) => x100(m.groups?.video_short?.acc), imageOnly: true },
      text: { title: say("Text", "文本"), get: (m) => x100(m.groups?.text?.acc), imageOnly: true },
      db_medium: { title: "DecisionBench medium", get: (m) => m.db_medium },
      db_hard: { title: "DecisionBench hard", get: (m) => m.db_hard },
      typesafe: { title: "TypeSafe", get: (m) => x100(m.typesafe_jev354) },
      mmbench: { title: "MMBench", get: (m) => x100(m.mmbench), imageOnly: true },
      mmstar: { title: "MMStar", get: (m) => x100(m.mmstar), imageOnly: true }
    };
    const val = (c, k) => {
      const v = metrics[k].get(c.m);
      return v == null || !isFinite(v) ? null : v;
    };
    const missing = (c, k) => c.textOnly && metrics[k].imageOnly ? say("text only", "仅文本") : "n/a";
    const box = $("#rpanels");
    box.replaceChildren(...["test", "db_hard", "typesafe", "mmstar"].map((k) => h(
      "div",
      { class: "rp" },
      h("h3", {}, metrics[k].title),
      cols.map((c) => {
        const v = val(c, k);
        return h("div", { class: `rp-row ${c.ours ? "ours" : "them"}` }, h("span", { class: "rl" }, c.label), blocks(v, 100, c.ours ? "" : "other", v == null ? missing(c, k) : v.toFixed(1)));
      })
    )));
    buildOnView(box);
    const rows = [["test", ""], ["gui", "sub"], ["image", "sub"], ["video_long", "sub"], ["video_short", "sub"], ["text", "sub"], ["db_medium", "rule"], ["db_hard", "rule"], ["typesafe", "rule"], ["mmbench", "rule"], ["mmstar", "rule"]];
    const t = $("#ptable");
    t.replaceChildren(
      h("thead", {}, h("tr", {}, h("th", {}, ""), cols.map((c) => h("th", { class: c.ours ? "th-ours" : "" }, c.head[0], h("br"), c.head[1])))),
      h("tbody", {}, rows.map(([k, cls]) => {
        const vs = cols.map((c) => val(c, k)), sorted = [...new Set(vs.filter((v) => v != null).map((v) => v.toFixed(1)))].map(Number).sort((a, b) => b - a);
        return h("tr", { class: cls }, h("td", {}, metrics[k].title), cols.map((c, j) => {
          const v = vs[j];
          if (v == null) return h("td", { class: "na" }, "n/a");
          const r = sorted.indexOf(Number(v.toFixed(1)));
          return h("td", { class: r === 0 ? "best" : r === 1 ? "second" : "" }, v.toFixed(1));
        }));
      }))
    );
    buildOnView(t.closest(".win"));
    const sp = $("#size-panel");
    const pairs = ["0.8B", "4B", "9B", "27B"].map((sz) => [models.find((m) => m.size === sz && !m.trained && m.status === "done"), models.find((m) => m.size === sz && m.trained && m.status === "done")]).filter(([a, b]) => a && b);
    sp.replaceChildren(h(
      "div",
      { class: "rp" },
      h("h3", {}, say("test accuracy, before and after", "测试准确率，训练前后")),
      pairs.flatMap(([a, b]) => [
        h("div", { class: "rp-row them" }, h("span", { class: "rl" }, a.name), blocks(a.test.acc * 100, 100, "other")),
        h("div", { class: "rp-row ours", style: "margin-bottom:10px" }, h("span", { class: "rl" }, disp(b.name)), blocks(b.test.acc * 100, 100, ""))
      ])
    ));
    buildOnView(sp);
    const pairSize = ["27B", "9B", "4B", "0.8B"].find((sz) => models.some((m) => m.size === sz && m.trained && m.status === "done"));
    const tr = models.find((m) => m.size === pairSize && m.trained && m.status === "done"), bs = models.find((m) => m.size === pairSize && !m.trained && m.status === "done");
    const gp = $("#group-panel");
    if (tr && bs) {
      gp.replaceChildren(h(
        "div",
        { class: "rp" },
        h("h3", {}, say(`${bs.name} and ${disp(tr.name)}`, `${bs.name} 和 ${disp(tr.name)}`)),
        GROUPS.flatMap(([k, l]) => [
          h("div", { class: "rp-row them" }, h("span", { class: "rl" }, say(`${l}, before`, `${l}，训练前`)), blocks(bs.groups[k].acc * 100, 100, "other")),
          h("div", { class: "rp-row ours", style: "margin-bottom:10px" }, h("span", { class: "rl" }, say(`${l}, after`, `${l}，训练后`)), blocks(tr.groups[k].acc * 100, 100, ""))
        ])
      ));
      buildOnView(gp);
    }
    const thinkExt = R.external.filter((x) => /generates text/.test(x.kind || ""));
    $("#think-text").textContent = say(`For scale, models that write out their reasoning before answering, on DecisionBench. The first three come from the benchmark card; Qwen3.8-27B in thinking mode is our run, with about 5,000 thinking tokens per state. OneJev answers in one forward pass and generates nothing.`, `作为参照，这里是先写出推理再作答的模型在 DecisionBench 上的成绩。前三个来自基准卡片；思考模式的 Qwen3.8-27B 是我们自己跑的，每个状态大约 5,000 个思考 token。OneJev 一次前向作答，不生成任何文字。`);
    $("#tbl-think").replaceChildren(
      h("thead", {}, h("tr", {}, h("th", {}, say("model", "模型")), h("th", {}, "DB medium"), h("th", {}, "DB hard"))),
      h("tbody", {}, thinkExt.map((x) => h("tr", {}, h("td", {}, x.name), h("td", {}, f1(x.db_medium)), h("td", {}, f1(x.db_hard)))))
    );
  }
  function mixTable(S) {
    if (!S.mix) return;
    const names = [["gui", say("GUI agents (web, desktop, phone)", "GUI agent（网页、桌面、手机）")], ["image", say("Images", "图片")], ["video_short", say("Short video", "短视频")], ["video_long", say("Long procedural video", "长流程视频")], ["text", say("Text agent runs", "文本 agent 运行")], ["rules", say("Business rules", "业务规则")]];
    $("#tbl-mix").replaceChildren(
      h("tr", {}, h("th", {}, say("part of the mix", "组成")), h("th", { class: "n" }, say("questions", "问题数")), h("th", { class: "n" }, say("sources", "来源数"))),
      ...names.filter(([k]) => S.mix.groups[k]).map(([k, l]) => h("tr", {}, h("td", {}, l), h("td", { class: "n" }, int(S.mix.groups[k].rows)), h("td", { class: "n" }, S.mix.groups[k].sources))),
      h("tr", {}, h("td", {}, h("b", {}, say("all", "合计"))), h("td", { class: "n" }, h("b", {}, int(S.mix.total))), h("td", { class: "n" }, ""))
    );
  }
  function localize() {
    if (!$("#os")) return;
    document.documentElement.lang = say("en", "zh-CN");
    if (ZH) {
      document.querySelectorAll("[data-zh]").forEach((e) => {
        e.textContent = e.dataset.zh;
      });
      document.querySelectorAll("[data-zh-html]").forEach((e) => {
        e.innerHTML = e.dataset.zhHtml;
      });
    }
    const other = ZH ? "en" : "zh", end = $(".top .end");
    if (end) end.append(h("a", { class: "lang", href: `?lang=${other}`, lang: ZH ? "en" : "zh-CN", onclick: (ev) => {
      ev.preventDefault();
      try {
        localStorage.setItem(LANG_KEY, other);
      } catch (e) {
      }
      const u = new URL(location.href);
      u.searchParams.set("lang", other);
      location.href = u.href;
    } }, ZH ? "EN" : "中文"));
  }
  async function main() {
    try {
      localize();
    } catch (e) {
      console.error(e);
    }
    try {
      eyesInit();
    } catch (e) {
      console.error(e);
    }
    if (!$("#os")) return;
    let R, S;
    try {
      [R, S] = await Promise.all([fetch("data/results.json", { cache: "no-cache" }).then((r) => r.json()), fetch("data/showcase.json", { cache: "no-cache" }).then((r) => r.json())]);
    } catch (e) {
      $("#news").textContent = say("Could not read data/results.json. Serve this folder over HTTP (python3 -m http.server) and reload.", "读不到 data/results.json。请用 HTTP 服务这个目录（python3 -m http.server）后刷新。");
      return;
    }
    const parts = [() => news(R), () => desktop(R, S), () => statements(R, S), () => streams(S), () => race(S), () => monitor(S), () => click(S), () => recipe(S), () => clips(S), () => photos(S), () => calibration(R, S), () => results(R), () => mixTable(S)];
    for (const p of parts) {
      try {
        await p();
      } catch (e) {
        console.error(e);
      }
    }
  }
  main();
})();
