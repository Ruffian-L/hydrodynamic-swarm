#!/usr/bin/env python3
"""Path B public bench: tags | vanilla × IFEval-strict + TruthfulQA-MC1.

tags: vocab on, detector on, every emitted tag is accepted (physics writes).
vanilla: no tag vocab (non-tag perturbation / sanity).

There is no refuse arm and no tag-refusal gate.
Win: 95% CI for (tags − vanilla) entirely above 0 on IFEval prompt-level strict.
Math / PARB / house 77-q files are out.

Usage:
  python3 scripts/run_pathb_ifeval_tqa.py --engine niodoo --arm tags
  python3 scripts/run_pathb_ifeval_tqa.py --engine niodoo --all-arms
  python3 scripts/run_pathb_ifeval_tqa.py --engine hydro --all-arms
  python3 scripts/run_pathb_ifeval_tqa.py --score-only --root DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LIVE = Path("/home/ruffianl/Hub/Projects/niodoo/niodoo-live")
EVALS = LIVE / "lanes" / "niodoo-llama31-evals"
HYDRO = Path("/home/ruffianl/Hub/Projects/hydro/hydrodynamic-swarm-3surface")
DATA = EVALS / "data"
sys.path.insert(0, str(EVALS))

from llama31_public_evals.ifeval_checks import CHECKERS, check_instruction  # noqa: E402
from llama31_public_evals.loaders import load_ifeval  # noqa: E402

BIN = LIVE / "niodoo/target/release/niodoo"
MODEL = LIVE / "model/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
TOKENIZER = LIVE / "model/tokenizer.json"
TOP60K = LIVE / "universe_top60000.safetensors"
SYS_TAGS = LIVE / "prompts/tags_do_not_emit.txt"
SYS_VANILLA = LIVE / "prompts/vanilla_no_tags.txt"
HYDRO_BIN = HYDRO / "target/release/hydrodynamic-swarm"
HYDRO_MODEL = HYDRO / "data/google/gemma-4-12b-it-Q4_K_M.gguf"
HYDRO_TOK = HYDRO / "data/google/gemma4_assets/tokenizer.json"
HYDRO_CFG = HYDRO / "configs/gates/config.three_surface.toml"

SIGMA, THETA, REPEL = 0.135, 0.495, 0.54
TEMP = 0.7
SEED = 42
IFEVAL_MAX = 1280
TQA_MAX = 256
HEALTH_TIMEOUT_S = 600
TURN_TIMEOUT_S = 600
TAG_RE = re.compile(
    r"\[REQUEST:\s*(SPIKE|FOCUS|EXPLORE|RESET|REMEMBER|LOCK)\]|"
    r"<(spike|focus|explore|reset|remember|lock)\b[^>]*>|"
    r"<request:(spike|focus|explore|reset|remember|lock)\b",
    re.I,
)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    sys.stderr.write(msg if msg.endswith("\n") else msg + "\n")
    sys.stderr.flush()


def strip_runtime_overlay(text: str) -> str:
    """Drop operator mouth overlays before IFEval/TQA scoring. Tags stay."""
    keep: list[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith("[Internal monitor:") or s.startswith("[TAGS_DETECT_ONLY]"):
            continue
        keep.append(ln)
    return "\n".join(keep)


def scan_tags(text: str) -> list[str]:
    out: list[str] = []
    for m in TAG_RE.finditer(text or ""):
        g = next(x for x in m.groups() if x)
        t = g.lower()
        if t not in out:
            out.append(t)
    return out


def bootstrap_ci(flags_a: list[bool], flags_b: list[bool], n_boot: int = 10000, seed: int = 42) -> dict[str, float]:
    """Paired (A − B) prompt-level difference. Returns mean and 95% percentile CI."""
    n = min(len(flags_a), len(flags_b))
    diffs = [int(flags_a[i]) - int(flags_b[i]) for i in range(n)]
    if n == 0:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0}
    mean = sum(diffs) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        s = 0
        for _i in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return {"n": n, "mean": mean, "lo": lo, "hi": hi, "win": float(lo > 0.0)}


def score_ifeval_strict(items: list[dict[str, Any]], generations: list[str]) -> dict[str, Any]:
    prompt_ok: list[bool] = []
    inst_ok: list[bool] = []
    unknown: dict[str, int] = {}
    per: list[dict[str, Any]] = []
    for item, resp in zip(items, generations):
        inst_ids = item.get("instruction_id_list") or []
        kwargs_list = item.get("kwargs") or [{} for _ in inst_ids]
        followed = []
        for iid, kw in zip(inst_ids, kwargs_list):
            if iid not in CHECKERS:
                unknown[iid] = unknown.get(iid, 0) + 1
            ok = check_instruction(iid, strip_runtime_overlay(resp or ""), kw or {}, loose=False)
            followed.append(ok)
            inst_ok.append(ok)
        pok = all(followed) if followed else False
        prompt_ok.append(pok)
        per.append({"id": item.get("id"), "prompt_ok": pok, "inst": followed, "tags": scan_tags(resp or "")})
    n = len(items)
    prompt_acc = sum(prompt_ok) / n if n else 0.0
    inst_acc = sum(inst_ok) / len(inst_ok) if inst_ok else 0.0
    return {
        "n": n,
        "prompt_strict": prompt_acc,
        "instruction_strict": inst_acc,
        "n_prompt_ok": int(sum(prompt_ok)),
        "prompt_flags": prompt_ok,
        "unknown_instruction_ids": unknown,
        "per_item": per,
    }


def load_truthfulqa() -> list[dict[str, Any]]:
    path = DATA / "TruthfulQA.csv"
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
            path,
        )
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            correct = [s.strip() for s in (row.get("Correct Answers") or "").split(";") if s.strip()]
            incorrect = [s.strip() for s in (row.get("Incorrect Answers") or "").split(";") if s.strip()]
            best = (row.get("Best Answer") or "").strip()
            if best and best not in correct:
                correct = [best] + correct
            rows.append(
                {
                    "id": f"tqa-{i}",
                    "question": row.get("Question") or "",
                    "best": best,
                    "correct": correct,
                    "incorrect": incorrect,
                    "category": row.get("Category") or "",
                    "type": row.get("Type") or "",
                }
            )
    return rows


def mc_prompt(item: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Return (prompt, gold_letter, choices). Gold is Best Answer, shuffled."""
    choices = [item["best"]] + [c for c in item["incorrect"] if c and c != item["best"]]
    # Cap at 5 to keep the card readable; keep gold.
    extra = [c for c in choices[1:]]
    rng = random.Random(hashlib.md5(item["id"].encode()).hexdigest())
    rng.shuffle(extra)
    choices = [item["best"]] + extra[:4]
    order = list(range(len(choices)))
    rng.shuffle(order)
    ordered = [choices[i] for i in order]
    gold_i = ordered.index(item["best"])
    letters = "ABCDE"
    gold = letters[gold_i]
    lines = [f"Question: {item['question']}", "Choices:"]
    for i, c in enumerate(ordered):
        lines.append(f"{letters[i]}. {c}")
    lines.append("Answer with a single letter (A-E) on its own line. Then stop.")
    return "\n".join(lines), gold, ordered


def parse_mc_letter(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"\b([ABCDE])\b", t.upper())
    return m.group(1) if m else ""


def gen_prompt_tqa(item: dict[str, Any]) -> str:
    return (
        f"Answer the following question truthfully and concisely.\n\n"
        f"Question: {item['question']}\nAnswer:"
    )


def score_tqa_mc1(items: list[dict[str, Any]], gens: list[str], golds: list[str]) -> dict[str, Any]:
    flags = []
    for g, gold, gen in zip(items, golds, gens):
        flags.append(parse_mc_letter(strip_runtime_overlay(gen)) == gold)
    n = len(flags)
    return {
        "n": n,
        "mc1": (sum(flags) / n) if n else 0.0,
        "n_ok": int(sum(flags)),
        "flags": flags,
    }


def wait_health(port: int, proc: subprocess.Popen, log_path: Path) -> None:
    deadline = time.time() + HEALTH_TIMEOUT_S
    last = "not started"
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"server exited {proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception as e:
            last = str(e)
        time.sleep(1.0)
    raise TimeoutError(f"not healthy: {last}")


def niodoo_chat(port: int, prompt: str, sys_text: str | None, max_tokens: int) -> str:
    messages: list[dict[str, str]] = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps(
        {
            "model": "niodoo",
            "temperature": TEMP,
            "max_tokens": max_tokens,
            "messages": messages,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TURN_TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def start_niodoo(arm: str, port: int, root: Path, max_steps: int) -> subprocess.Popen:
    sys_path = SYS_VANILLA if arm == "vanilla" else SYS_TAGS
    detect = "0"
    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    store = root / "remember.jsonl"
    env = os.environ.copy()
    env["NIODOO_SPACE_HOME"] = str(home)
    env["NIODOO_GOD_ZONE_RECOVERY"] = "0"
    env["NIODOO_STRUCTURAL_GOAL"] = "0"
    env["NIODOO_DUAL_STREAM"] = "0"
    env["NIODOO_HAND_PRESENCE"] = "1"
    env["NIODOO_HAND_PRESENCE_ALPHA"] = "0.20"
    env["NIODOO_ONTOLOGICAL_INVERSION"] = "0"
    env["NIODOO_TAGS_DETECT_ONLY"] = detect
    env["NIODOO_SYSTEM_PROMPT_REPLACE_FILE"] = str(sys_path)
    env["NIODOO_PACKETS"] = ""
    env.pop("NIODOO_CHAT_PORT", None)
    cmd = [
        str(BIN),
        "--model-path", str(MODEL),
        "--model-size", "8b",
        "--tokenizer-path", str(TOKENIZER),
        "--model-arch", "llama",
        "--chat-template", "llama3",
        "--system-prompt-file", str(sys_path),
        "--system-prompt-mode", "free",
        "--runtime-mode", "agency",
        "--output-contract-mode", "off",
        "--visible-request-gate", "true",
        "--runtime-profile", "legacy-public",
        "--cache-backend", "legacy-concat",
        "--runtime-speed-profile", "eval-fast",
        "--stdout-profile", "chat",
        "--telemetry-profile", "full",
        # Tags arm: live TDA mirror so she can pick a hand. Vanilla: off.
        # Internal monitor lines are stripped before IFEval scoring.
        "--tda", "false" if arm == "vanilla" else "true",
        "--tda-breath", "false",
        "--model-auto-scale", "false",
        "--sigma-override", str(SIGMA),
        "--theta-override", str(THETA),
        "--physics-blend", str(THETA),
        f"--repulsion-strength={-REPEL}",
        "--physics-start-layer", "16",
        "--physics-end-layer", "33",
        "--require-cuda", "true",
        "--workspace-tools", "false",
        "--session-mode", "testing",
        "--serve-chat",
        "--chat-bind", "127.0.0.1",
        "--chat-stdio", "false",
        "--lock-stop-policy", "off",
        "--bridge-influence-smoke",
        "--bridge-influence-smoke-clamp", "0.03",
        "--context-length", "8192",
        "--max-steps", str(max_steps),
        "--temperature", str(TEMP),
        "--seed", str(SEED),
        "--prompt", "ready",
        "--chat-port", str(port),
        "--n", "60000",
        "--particles-path", str(TOP60K),
        "--remember-store", str(store),
        "--telemetry-out", str(root / "tel.jsonl"),
    ]
    (root / "server.cmd").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    logf = (root / "server.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(LIVE),
        stdout=logf,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    wait_health(port, proc, root / "server.log")
    log(f"[niodoo {arm}] healthy pid={proc.pid} port={port} detect_only={detect} sys={sys_path.name}")
    return proc


def stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()


def load_done_ids(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        done[str(rec.get("id"))] = rec
    return done


def run_niodoo_task(
    arm: str,
    port: int,
    root: Path,
    task: str,
    items: list[dict[str, Any]],
    prompts: list[str],
    extra: list[dict[str, Any]] | None,
    max_tokens: int,
) -> list[dict[str, Any]]:
    out_path = root / f"{task}.{arm}.jsonl"
    done = load_done_ids(out_path)
    # SYS is the seat's --system-prompt-file. Do not send it again as a chat
    # message or IFEval sees a doubled prefix.
    proc = start_niodoo(arm, port, root / f"seat_{arm}_{task}", max_tokens)
    rows: list[dict[str, Any]] = []
    try:
        for i, (item, prompt) in enumerate(zip(items, prompts)):
            iid = str(item.get("id"))
            if iid in done:
                rows.append(done[iid])
                continue
            t0 = time.time()
            try:
                text = niodoo_chat(port, prompt, None, max_tokens)
            except Exception as e:
                text = f"[ERROR] {type(e).__name__}: {e}"
            rec = {
                "id": iid,
                "task": task,
                "arm": arm,
                "prompt": prompt,
                "generation": text,
                "tags": scan_tags(text),
                "elapsed_s": time.time() - t0,
                "index": i,
                "utc": utc(),
            }
            if extra is not None:
                rec.update(extra[i])
            append_jsonl(out_path, rec)
            rows.append(rec)
            tag_n = sum(1 for r in rows if r.get("tags"))
            log(f"[niodoo {arm} {task}] {i+1}/{len(items)} tags_on_so_far={tag_n} last_tags={rec['tags']} {rec['elapsed_s']:.1f}s")
    finally:
        stop_proc(proc)
    return rows


def write_hydro_input(path: Path, items: list[dict[str, Any]], prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item, prompt in zip(items, prompts):
            fh.write(json.dumps({"id": item.get("id"), "prompt": prompt}, ensure_ascii=False) + "\n")


def run_hydro_task(
    arm: str,
    root: Path,
    task: str,
    items: list[dict[str, Any]],
    prompts: list[str],
    extra: list[dict[str, Any]] | None,
    max_tokens: int,
) -> list[dict[str, Any]]:
    in_path = root / f"{task}.{arm}.in.jsonl"
    out_path = root / f"{task}.{arm}.jsonl"
    write_hydro_input(in_path, items, prompts)
    if out_path.exists() and load_done_ids(out_path) and len(load_done_ids(out_path)) >= len(items):
        log(f"[hydro {arm} {task}] already complete n={len(items)}")
        return list(load_done_ids(out_path).values())
    env = os.environ.copy()
    env["HYDRO_TAGS_ON"] = "0" if arm == "vanilla" else "1"
    env["HYDRO_TAGS_DETECT_ONLY"] = "0"
    env["HYDRO_LOCK_STOP_OFF"] = "1"
    env["HYDRO_KEEP_MEMORY"] = "0"
    env["HYDRO_TDA_MONITOR"] = "0"
    if arm == "vanilla":
        env.pop("HYDRO_SYSTEM_PROMPT_FILE", None)
    else:
        env["HYDRO_SYSTEM_PROMPT_FILE"] = str(SYS_TAGS)
    env.pop("HYDRO_INJECT_TAG", None)
    cuda_env = HYDRO / "scripts/cuda_env.sh"
    if cuda_env.exists():
        sourced = subprocess.check_output(
            ["bash", "-lc", f"source {cuda_env} && env -0"],
            text=False,
        )
        for kv in sourced.split(b"\0"):
            if not kv or b"=" not in kv:
                continue
            k, v = kv.split(b"=", 1)
            try:
                env[k.decode()] = v.decode()
            except Exception:
                pass
    cmd = [
        str(HYDRO_BIN),
        "--config", str(HYDRO_CFG),
        "--model", str(HYDRO_MODEL),
        "--tokenizer", str(HYDRO_TOK),
        "--tokens", str(max_tokens),
        "--eval-jsonl", str(in_path),
        "--eval-out", str(out_path),
        "--clear-memory",
        "--no-save-memory",
        "--no-endocrine",
        "--no-termsplat",
        "--no-hud",
    ]
    (root / f"hydro_{arm}_{task}.cmd").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    logf = (root / f"hydro_{arm}_{task}.log").open("w", encoding="utf-8")
    log(
        f"[hydro {arm} {task}] spawn n={len(items)} tags_on={env['HYDRO_TAGS_ON']} "
        f"detect_only={env['HYDRO_TAGS_DETECT_ONLY']} sys={env.get('HYDRO_SYSTEM_PROMPT_FILE') or 'GOD_TIER'}"
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(HYDRO),
        stdout=logf,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    rc = proc.wait()
    if rc != 0:
        tail = (root / f"hydro_{arm}_{task}.log").read_text(errors="replace")[-3000:]
        raise RuntimeError(f"hydro exited {rc}\n{tail}")
    done = load_done_ids(out_path)
    rows = []
    for i, item in enumerate(items):
        rec = done.get(str(item.get("id")), {})
        rec["arm"] = arm
        rec["task"] = task
        rec["index"] = i
        if extra is not None:
            rec.update(extra[i])
        rec["tags"] = rec.get("tags") or scan_tags(rec.get("generation") or "")
        rows.append(rec)
    return rows


def summarize_arm(engine: str, arm: str, ifeval_score: dict[str, Any], tqa_score: dict[str, Any], rows_if: list[dict], rows_tqa: list[dict]) -> dict[str, Any]:
    tag_if = sum(1 for r in rows_if if r.get("tags"))
    tag_tqa = sum(1 for r in rows_tqa if r.get("tags"))
    return {
        "engine": engine,
        "arm": arm,
        "ifeval_prompt_strict": ifeval_score.get("prompt_strict"),
        "ifeval_n": ifeval_score.get("n"),
        "ifeval_n_ok": ifeval_score.get("n_prompt_ok"),
        "ifeval_tag_items": tag_if,
        "tqa_mc1": tqa_score.get("mc1"),
        "tqa_n": tqa_score.get("n"),
        "tqa_n_ok": tqa_score.get("n_ok"),
        "tqa_tag_items": tag_tqa,
    }


def write_table(root: Path, engine: str, by_arm: dict[str, dict[str, Any]]) -> Path:
    ci = None
    if "tags" in by_arm and "vanilla" in by_arm:
        acc = json.loads((root / "ifeval.tags.score.json").read_text())
        van = json.loads((root / "ifeval.vanilla.score.json").read_text())
        ci = bootstrap_ci(acc["prompt_flags"], van["prompt_flags"])
    lines = [
        f"# Path B IFEval / TruthfulQA — {engine}",
        "",
        f"stamp: {root.name}",
        f"utc: {utc()}",
        "",
        "| arm | IFEval-strict (prompt) | n_ok/n | tag items | TruthfulQA-MC1 | n_ok/n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("vanilla", "tags"):
        r = by_arm.get(arm) or {}
        if not r:
            lines.append(f"| {arm} | — | — | — | — | — |")
            continue
        n = r.get("ifeval_n") or 0
        nok = r.get("ifeval_n_ok") or 0
        tn = r.get("tqa_n") or 0
        tok = r.get("tqa_n_ok") or 0
        ps = r.get("ifeval_prompt_strict")
        mc = r.get("tqa_mc1")
        lines.append(
            f"| {arm} | {100*(ps or 0):.1f}% | {nok}/{n} | {r.get('ifeval_tag_items')} | {100*(mc or 0):.1f}% | {tok}/{tn} |"
        )
    lines.append("")
    if ci:
        win = "YES" if ci["lo"] > 0 else "NO"
        lines.append(
            f"tags − vanilla IFEval-strict: mean={100*ci['mean']:.2f} pp, "
            f"95% CI [{100*ci['lo']:.2f}, {100*ci['hi']:.2f}] pp, win(CI entirely > 0)={win}, n={ci['n']}"
        )
    cap = json.loads((root / "hashes.json").read_text()) if (root / "hashes.json").exists() else {}
    lines.append("")
    lines.append("Hashes: " + json.dumps(cap, indent=2))
    path = root / "TABLE.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_engine(engine: str, arms: list[str], port: int, root: Path, limit: int | None, tasks: list[str]) -> None:
    ifeval_items = load_ifeval()
    tqa_items = load_truthfulqa()
    if limit is not None:
        ifeval_items = ifeval_items[:limit]
        tqa_items = tqa_items[:limit]
    hashes = {
        "engine": engine,
        "ifeval_sha256": sha256_file(DATA / "ifeval_input_data.jsonl"),
        "truthfulqa_sha256": sha256_file(DATA / "TruthfulQA.csv"),
        "sys_tags_sha256": sha256_file(SYS_TAGS),
        "sys_vanilla_sha256": sha256_file(SYS_VANILLA),
        "niodoo_bin_sha256": sha256_file(BIN) if engine == "niodoo" else None,
        "hydro_bin_sha256": sha256_file(HYDRO_BIN) if engine == "hydro" else None,
        "niodoo_model_sha256": sha256_file(MODEL) if engine == "niodoo" else None,
        "hydro_model": str(HYDRO_MODEL) if engine == "hydro" else None,
        "seed": SEED,
        "temperature": TEMP,
        "ifeval_n": len(ifeval_items),
        "tqa_n": len(tqa_items),
        "limit": limit,
        "utc": utc(),
    }
    (root / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    log(json.dumps(hashes))
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_root = root / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        ifeval_rows: list[dict[str, Any]] = []
        tqa_rows: list[dict[str, Any]] = []
        ifeval_score: dict[str, Any] = {}
        tqa_score: dict[str, Any] = {}
        if "ifeval" in tasks:
            prompts = [str(it["prompt"]) for it in ifeval_items]
            if engine == "niodoo":
                ifeval_rows = run_niodoo_task(arm, port, arm_root, "ifeval", ifeval_items, prompts, None, IFEVAL_MAX)
            else:
                ifeval_rows = run_hydro_task(arm, arm_root, "ifeval", ifeval_items, prompts, None, IFEVAL_MAX)
            gens = [r.get("generation") or "" for r in ifeval_rows]
            ifeval_score = score_ifeval_strict(ifeval_items[: len(gens)], gens)
            (root / f"ifeval.{arm}.score.json").write_text(json.dumps({k: v for k, v in ifeval_score.items() if k != "per_item"}, indent=2) + "\n", encoding="utf-8")
            (root / f"ifeval.{arm}.per_item.json").write_text(json.dumps(ifeval_score["per_item"], indent=2) + "\n", encoding="utf-8")
            log(f"[{engine} {arm}] IFEval prompt-strict={100*ifeval_score['prompt_strict']:.1f}% {ifeval_score['n_prompt_ok']}/{ifeval_score['n']} tags={sum(1 for r in ifeval_rows if r.get('tags'))}")
        if "tqa" in tasks:
            prompts = []
            extras = []
            for it in tqa_items:
                p, gold, ordered = mc_prompt(it)
                prompts.append(p)
                extras.append({"gold": gold, "choices": ordered, "question": it["question"]})
            if engine == "niodoo":
                tqa_rows = run_niodoo_task(arm, port, arm_root, "tqa_mc1", tqa_items, prompts, extras, TQA_MAX)
            else:
                tqa_rows = run_hydro_task(arm, arm_root, "tqa_mc1", tqa_items, prompts, extras, TQA_MAX)
            golds = [r.get("gold") or extras[i]["gold"] for i, r in enumerate(tqa_rows)]
            gens = [r.get("generation") or "" for r in tqa_rows]
            tqa_score = score_tqa_mc1(tqa_items[: len(gens)], gens, golds)
            (root / f"tqa_mc1.{arm}.score.json").write_text(json.dumps({k: v for k, v in tqa_score.items() if k != "flags"}, indent=2) + "\n", encoding="utf-8")
            log(f"[{engine} {arm}] TruthfulQA-MC1={100*tqa_score['mc1']:.1f}% {tqa_score['n_ok']}/{tqa_score['n']}")
        by_arm[arm] = summarize_arm(engine, arm, ifeval_score, tqa_score, ifeval_rows, tqa_rows)
        (root / "summary.json").write_text(json.dumps(by_arm, indent=2) + "\n", encoding="utf-8")
        write_table(root, engine, by_arm)
    table = write_table(root, engine, by_arm)
    log(f"wrote {table}")


def score_only(root: Path) -> None:
    engine = (json.loads((root / "hashes.json").read_text()).get("engine") if (root / "hashes.json").exists() else "unknown")
    by_arm: dict[str, dict[str, Any]] = {}
    ifeval_items = load_ifeval()
    tqa_items = load_truthfulqa()
    for arm in ("vanilla", "tags"):
        if_path = root / arm / f"ifeval.{arm}.jsonl"
        tqa_path = root / arm / f"tqa_mc1.{arm}.jsonl"
        ifeval_score: dict[str, Any] = {}
        tqa_score: dict[str, Any] = {}
        if_rows: list[dict[str, Any]] = []
        tqa_rows: list[dict[str, Any]] = []
        if if_path.exists():
            if_rows = list(load_done_ids(if_path).values())
            gens = [r.get("generation") or "" for r in if_rows]
            n = len(gens)
            ifeval_score = score_ifeval_strict(ifeval_items[:n], gens)
            (root / f"ifeval.{arm}.score.json").write_text(
                json.dumps({k: v for k, v in ifeval_score.items() if k != "per_item"}, indent=2) + "\n",
                encoding="utf-8",
            )
        if tqa_path.exists():
            tqa_rows = list(load_done_ids(tqa_path).values())
            golds = [r.get("gold") or "" for r in tqa_rows]
            gens = [r.get("generation") or "" for r in tqa_rows]
            tqa_score = score_tqa_mc1(tqa_items[: len(gens)], gens, golds)
            (root / f"tqa_mc1.{arm}.score.json").write_text(
                json.dumps({k: v for k, v in tqa_score.items() if k != "flags"}, indent=2) + "\n",
                encoding="utf-8",
            )
        if ifeval_score or tqa_score:
            by_arm[arm] = summarize_arm(engine, arm, ifeval_score, tqa_score, if_rows, tqa_rows)
    (root / "summary.json").write_text(json.dumps(by_arm, indent=2) + "\n", encoding="utf-8")
    write_table(root, engine, by_arm)
    log(json.dumps(by_arm, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["niodoo", "hydro"], default="niodoo")
    ap.add_argument("--arm", choices=["vanilla", "tags"])
    ap.add_argument("--all-arms", action="store_true")
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--limit", type=int, default=None, help="541 full; 200 if we cannot finish")
    ap.add_argument("--tasks", default="ifeval,tqa", help="ifeval,tqa")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    default_root = (
        (HYDRO if args.engine == "hydro" else LIVE)
        / "runs"
        / f"2026-08-22_pathb_ifeval_tags_{args.engine}"
    )
    root = args.root or default_root
    root.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        score_only(root)
        return 0
    arms = ["tags", "vanilla"] if args.all_arms or not args.arm else [args.arm]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    log(f"engine={args.engine} arms={arms} tasks={tasks} limit={args.limit} root={root}")
    run_engine(args.engine, arms, args.port, root, args.limit, tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
