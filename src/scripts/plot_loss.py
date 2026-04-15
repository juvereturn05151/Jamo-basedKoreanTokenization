import json
from pathlib import Path
import matplotlib.pyplot as plt

RUNS = Path(__file__).resolve().parents[2] / "runs"
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(exist_ok=True)

SMOOTH = 0.9


def ema(values, alpha=SMOOTH):
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * acc + (1 - alpha) * v
        out.append(acc)
    return out


def latest_log(tag: str) -> Path:
    logs = sorted((RUNS / tag).glob("*/log.jsonl"))
    if not logs:
        raise FileNotFoundError(f"no logs under runs/{tag}")
    return logs[-1]


def load(path: Path):
    steps, losses = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("type") == "step":
            steps.append(rec["step"])
            losses.append(rec["loss"])
    return steps, losses


DPI = 300
LW_RAW = 0.5
LW_SMOOTH = 0.9


def _save(out: Path):
    plt.savefig(out, dpi=DPI, transparent=True)
    plt.close()
    print(f"saved {out}")


def plot_single(tag: str, color: str, smooth: bool):
    path = latest_log(tag)
    steps, losses = load(path)
    plt.figure(figsize=(8, 5))
    if smooth:
        plt.plot(steps, losses, color=color, alpha=0.25, linewidth=LW_RAW)
        plt.plot(steps, ema(losses), color=color, linewidth=LW_SMOOTH, label=f"{tag} (ema {SMOOTH})")
    else:
        plt.plot(steps, losses, color=color, linewidth=LW_RAW, label=tag)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"{tag} — {path.parent.name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _save(OUT / (f"{tag}_smooth.png" if smooth else f"{tag}.png"))
    return steps, losses


def plot_combined(full, jamo, smooth: bool):
    plt.figure(figsize=(8, 5))
    if smooth:
        plt.plot(full[0], full[1], color="tab:blue", alpha=0.2, linewidth=LW_RAW)
        plt.plot(jamo[0], jamo[1], color="tab:red", alpha=0.2, linewidth=LW_RAW)
        plt.plot(full[0], ema(full[1]), color="tab:blue", linewidth=LW_SMOOTH, label="hf_full")
        plt.plot(jamo[0], ema(jamo[1]), color="tab:red", linewidth=LW_SMOOTH, label="hf_jamo")
        title = f"hf_full vs hf_jamo (ema {SMOOTH})"
        out = OUT / "combined_smooth.png"
    else:
        plt.plot(full[0], full[1], color="tab:blue", linewidth=LW_RAW, label="hf_full")
        plt.plot(jamo[0], jamo[1], color="tab:red", linewidth=LW_RAW, label="hf_jamo")
        title = "hf_full vs hf_jamo"
        out = OUT / "combined.png"
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _save(out)


if __name__ == "__main__":
    for smooth in (False, True):
        full = plot_single("hf_full", "tab:blue", smooth)
        jamo = plot_single("hf_jamo", "tab:red", smooth)
        plot_combined(full, jamo, smooth)
