import argparse
import csv
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def db20(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return 20.0 * torch.log10(torch.clamp(x, min=eps))


def normalized_db(af: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return db20(af / (torch.amax(af, dim=1, keepdim=True) + eps), eps)


def build_planar_positions(rows: int, cols: int, spacing: float, device: torch.device) -> torch.Tensor:
    xs = (torch.arange(cols, dtype=torch.float32, device=device) - (cols - 1) / 2.0) * spacing
    ys = (torch.arange(rows, dtype=torch.float32, device=device) - (rows - 1) / 2.0) * spacing
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def build_taper(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    wx = torch.blackman_window(cols, periodic=False, dtype=torch.float32, device=device)
    wy = torch.blackman_window(rows, periodic=False, dtype=torch.float32, device=device)
    taper = torch.outer(wy, wx).reshape(1, rows * cols)
    return taper / torch.amax(taper)


def steering_phase(positions: torch.Tensor, theta_deg: float, phi_deg: float) -> torch.Tensor:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    u = math.sin(theta) * math.cos(phi)
    v = math.sin(theta) * math.sin(phi)
    return 2.0 * math.pi * (positions[:, 0] * u + positions[:, 1] * v)


def array_factor(weights: torch.Tensor, positions: torch.Tensor, theta_rad: torch.Tensor, phi_deg: float) -> torch.Tensor:
    phi = math.radians(phi_deg)
    u = torch.sin(theta_rad) * math.cos(phi)
    v = torch.sin(theta_rad) * math.sin(phi)
    phase = 2.0 * math.pi * (positions[:, 0:1] * u[None, :] + positions[:, 1:2] * v[None, :])
    steering = torch.exp(1j * phase)
    return torch.abs(torch.sum(weights[:, :, None] * steering[None, :, :], dim=1))


class CompetitionPINAS(nn.Module):
    """Physics-informed MLP residual model for sum and difference beam synthesis."""

    def __init__(self, element_count: int, hidden: int = 512):
        super().__init__()
        self.element_count = element_count
        self.net = nn.Sequential(
            nn.Linear(8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 * element_count),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        base_taper: torch.Tensor,
        base_steering: torch.Tensor,
        diff_sign: torch.Tensor,
        fault_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        n = self.element_count
        sum_amp_res = 0.50 + torch.sigmoid(raw[:, :n])
        sum_phase_res = 0.35 * math.pi * torch.tanh(raw[:, n : 2 * n])
        diff_amp_res = 0.50 + torch.sigmoid(raw[:, 2 * n : 3 * n])
        diff_phase_res = 0.35 * math.pi * torch.tanh(raw[:, 3 * n : 4 * n])

        steer = torch.exp(-1j * base_steering)[None, :]
        sum_weights = base_taper * sum_amp_res * torch.exp(1j * sum_phase_res) * steer
        diff_weights = base_taper * diff_sign[None, :] * diff_amp_res * torch.exp(1j * diff_phase_res) * steer
        return sum_weights * fault_mask, diff_weights * fault_mask, sum_amp_res, diff_amp_res


def parse_nulls(text: str, theta0: float | None = None, guard_deg: float = 12.0) -> list[float]:
    if not text.strip():
        return []
    if text.strip().lower() == "auto4":
        if theta0 is None:
            theta0 = 0.0
        candidates = [-60.0, -45.0, -30.0, -15.0, 15.0, 30.0, 45.0, 60.0]
        valid = [angle for angle in candidates if abs(angle - theta0) > guard_deg]
        valid.sort(key=lambda angle: (abs(angle - theta0), abs(angle)))
        return valid[:4]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def make_fault_mask(element_count: int, fault_rate: float, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    keep = torch.rand((1, element_count), generator=generator, device=device) >= fault_rate
    return keep.to(torch.complex64)


def topk_excess_loss(values_db: torch.Tensor, limit_db: float, topk: int) -> torch.Tensor:
    excess = torch.relu(values_db - limit_db)
    k = min(topk, excess.shape[1])
    return torch.mean(torch.topk(excess, k=k, dim=1).values ** 2)


def effective_guard(base_guard_deg: float, theta0: float) -> float:
    scan_projection = max(abs(math.cos(math.radians(theta0))), 0.45)
    return base_guard_deg / scan_projection


def compute_losses(
    sum_db: torch.Tensor,
    diff_db: torch.Tensor,
    theta_deg: torch.Tensor,
    theta0: float,
    nulls: list[float],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    sum_guard = effective_guard(args.sum_guard_deg, theta0)
    diff_guard = effective_guard(args.diff_guard_deg, theta0)
    target_idx = int(torch.argmin(torch.abs(theta_deg - theta0)).item())
    sum_main = sum_db[:, target_idx]
    sum_side = sum_db[:, torch.abs(theta_deg - theta0) > sum_guard]
    loss_sum_point = torch.mean((-sum_main) ** 2)
    loss_sum_sll = topk_excess_loss(sum_side, args.sum_sll_db, args.sll_topk)

    loss_null = torch.zeros((), dtype=torch.float32, device=sum_db.device)
    for angle in nulls:
        idx = int(torch.argmin(torch.abs(theta_deg - angle)).item())
        loss_null = loss_null + torch.mean(torch.relu(sum_db[:, idx] - args.null_depth_db) ** 2)

    diff_center = diff_db[:, target_idx]
    diff_side = diff_db[:, torch.abs(theta_deg - theta0) > diff_guard]
    lobe_left = int(torch.argmin(torch.abs(theta_deg - (theta0 - args.diff_lobe_offset_deg))).item())
    lobe_right = int(torch.argmin(torch.abs(theta_deg - (theta0 + args.diff_lobe_offset_deg))).item())
    loss_diff_null = torch.mean(torch.relu(diff_center - args.diff_null_db) ** 2)
    loss_diff_lobes = torch.mean((-diff_db[:, lobe_left]) ** 2 + (-diff_db[:, lobe_right]) ** 2)
    loss_diff_sll = topk_excess_loss(diff_side, args.diff_sll_db, args.sll_topk)

    loss = (
        args.w_sum_point * loss_sum_point
        + args.w_sum_sll * loss_sum_sll
        + args.w_null * loss_null
        + args.w_diff_null * loss_diff_null
        + args.w_diff_lobes * loss_diff_lobes
        + args.w_diff_sll * loss_diff_sll
    )
    metrics = {
        "loss_sum_point": float(loss_sum_point.detach().cpu()),
        "loss_sum_sll": float(loss_sum_sll.detach().cpu()),
        "loss_null": float(loss_null.detach().cpu()),
        "loss_diff_null": float(loss_diff_null.detach().cpu()),
        "loss_diff_lobes": float(loss_diff_lobes.detach().cpu()),
        "loss_diff_sll": float(loss_diff_sll.detach().cpu()),
    }
    return loss, metrics


def evaluate(theta_deg: torch.Tensor, sum_db: torch.Tensor, diff_db: torch.Tensor, theta0: float, nulls: list[float], args: argparse.Namespace) -> dict[str, float]:
    angles = theta_deg.detach().cpu()
    s = sum_db[0].detach().cpu()
    d = diff_db[0].detach().cpu()
    sum_peak_idx = int(torch.argmax(s).item())
    sum_guard = effective_guard(args.sum_guard_deg, theta0)
    diff_guard = effective_guard(args.diff_guard_deg, theta0)
    sum_side = s[torch.abs(angles - theta0) > sum_guard]
    diff_side = d[torch.abs(angles - theta0) > diff_guard]
    target_idx = int(torch.argmin(torch.abs(angles - theta0)).item())
    out = {
        "theta0_deg": theta0,
        "elements": args.rows * args.cols,
        "scan_limit_deg": args.scan_limit_deg,
        "sum_guard_used_deg": sum_guard,
        "diff_guard_used_deg": diff_guard,
        "sum_peak_theta_deg": float(angles[sum_peak_idx].item()),
        "sum_pointing_error_deg": abs(float(angles[sum_peak_idx].item()) - theta0),
        "sum_target_gain_db": float(s[target_idx].item()),
        "sum_sll_db": float(torch.amax(sum_side).item()),
        "diff_null_at_target_db": float(d[target_idx].item()),
        "diff_sll_db": float(torch.amax(diff_side).item()),
    }
    for idx, angle in enumerate(nulls, start=1):
        null_idx = int(torch.argmin(torch.abs(angles - angle)).item())
        out[f"null_{idx}_theta_deg"] = angle
        out[f"null_{idx}_depth_db"] = float(s[null_idx].item())
    return out


def save_plot(path: Path, theta_deg: torch.Tensor, sum_db: torch.Tensor, diff_db: torch.Tensor, theta0: float, nulls: list[float], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    angles = theta_deg.detach().cpu().numpy()
    s = sum_db[0].detach().cpu().numpy()
    d = diff_db[0].detach().cpu().numpy()
    plt.figure(figsize=(11, 6))
    plt.plot(angles, s, label="Sum beam", linewidth=2.0)
    plt.plot(angles, d, label="Difference beam", linewidth=1.7)
    plt.axvline(theta0, color="tab:red", linestyle="--", linewidth=1.2, label="Target")
    plt.axhline(args.sum_sll_db, color="tab:green", linestyle="--", linewidth=1.0, label="Sum SLL limit")
    plt.axhline(args.diff_sll_db, color="tab:orange", linestyle="--", linewidth=1.0, label="Diff SLL limit")
    for angle in nulls:
        plt.axvline(angle, color="tab:purple", linestyle=":", linewidth=1.0)
    plt.ylim(-80, 3)
    plt.xlim(args.theta_min, args.theta_max)
    plt.grid(True, alpha=0.28)
    plt.xlabel("Theta (deg), phi=0 deg")
    plt.ylabel("Normalized pattern (dB)")
    plt.title(f"Competition PI-NAS {args.rows}x{args.cols}, theta0={theta0:g} deg")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_weights(path: Path, sum_weights: torch.Tensor, diff_weights: torch.Tensor, rows: int, cols: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sw = sum_weights[0].detach().cpu()
    dw = diff_weights[0].detach().cpu()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "sum_amp", "sum_phase_deg", "diff_amp", "diff_phase_deg"])
        for idx in range(rows * cols):
            writer.writerow([
                idx // cols,
                idx % cols,
                f"{abs(complex(sw[idx])):.8f}",
                f"{math.degrees(math.atan2(float(sw[idx].imag), float(sw[idx].real))):.6f}",
                f"{abs(complex(dw[idx])):.8f}",
                f"{math.degrees(math.atan2(float(dw[idx].imag), float(dw[idx].real))):.6f}",
            ])


def save_patterns(path: Path, theta_deg: torch.Tensor, sum_db: torch.Tensor, diff_db: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_deg", "sum_db", "diff_db"])
        for theta, s, d in zip(theta_deg.detach().cpu().tolist(), sum_db[0].detach().cpu().tolist(), diff_db[0].detach().cpu().tolist()):
            writer.writerow([f"{theta:.4f}", f"{s:.6f}", f"{d:.6f}"])


def train_case(args: argparse.Namespace, theta0: float) -> dict[str, float]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    positions = build_planar_positions(args.rows, args.cols, args.spacing, device)
    element_count = args.rows * args.cols
    base_taper = build_taper(args.rows, args.cols, device)
    base_phase = steering_phase(positions, theta0, args.phi0)
    diff_sign = torch.sign(positions[:, 0] - torch.mean(positions[:, 0]))
    diff_sign[diff_sign == 0] = 1.0
    fault_mask = make_fault_mask(element_count, args.fault_rate, device, args.seed)
    theta_deg = torch.linspace(args.theta_min, args.theta_max, args.theta_points, device=device)
    theta_rad = torch.deg2rad(theta_deg)
    nulls = parse_nulls(args.nulls, theta0, effective_guard(args.sum_guard_deg, theta0))

    model = CompetitionPINAS(element_count, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.tensor(
        [[theta0 / 90.0, args.phi0 / 180.0, args.sum_sll_db / 60.0, args.diff_sll_db / 60.0,
          args.null_depth_db / 60.0, args.fault_rate, args.rows / 64.0, args.cols / 64.0]],
        dtype=torch.float32,
        device=device,
    )

    start = time.perf_counter()
    last_loss = {}
    for epoch in range(1, args.epochs + 1):
        sum_w, diff_w, _, _ = model(x, base_taper, base_phase, diff_sign, fault_mask)
        sum_db = normalized_db(array_factor(sum_w, positions, theta_rad, args.phi0))
        diff_db = normalized_db(array_factor(diff_w, positions, theta_rad, args.phi0))
        loss, last_loss = compute_losses(sum_db, diff_db, theta_deg, theta0, nulls, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            metrics = evaluate(theta_deg, sum_db, diff_db, theta0, nulls, args)
            print(
                f"theta0={theta0:>6.1f} epoch={epoch:>5d} loss={loss.item():.4f} "
                f"sum_peak={metrics['sum_peak_theta_deg']:.2f} sum_sll={metrics['sum_sll_db']:.2f}dB "
                f"diff_null={metrics['diff_null_at_target_db']:.2f}dB"
            )

    sum_w, diff_w, _, _ = model(x, base_taper, base_phase, diff_sign, fault_mask)
    sum_db = normalized_db(array_factor(sum_w, positions, theta_rad, args.phi0))
    diff_db = normalized_db(array_factor(diff_w, positions, theta_rad, args.phi0))
    metrics = evaluate(theta_deg, sum_db, diff_db, theta0, nulls, args)
    metrics.update(last_loss)
    metrics["train_seconds"] = time.perf_counter() - start
    metrics["fault_rate"] = args.fault_rate

    stem = f"competition_theta_{theta0:+.0f}".replace("+", "p").replace("-", "m")
    out_dir = Path(args.output_dir)
    save_plot(out_dir / f"{stem}.png", theta_deg, sum_db, diff_db, theta0, nulls, args)
    save_patterns(out_dir / f"{stem}_patterns.csv", theta_deg, sum_db, diff_db)
    save_weights(out_dir / f"{stem}_weights.csv", sum_w, diff_w, args.rows, args.cols)
    return metrics


def write_summary(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Competition-oriented PI-NAS array synthesis.")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--theta0", type=float, default=30.0)
    parser.add_argument("--phi0", type=float, default=0.0)
    parser.add_argument("--sweep", type=float, nargs="*", default=None)
    parser.add_argument("--theta-min", type=float, default=-90.0)
    parser.add_argument("--theta-max", type=float, default=90.0)
    parser.add_argument("--theta-points", type=int, default=721)
    parser.add_argument("--scan-limit-deg", type=float, default=60.0)
    parser.add_argument("--sum-sll-db", type=float, default=-35.0)
    parser.add_argument("--diff-null-db", type=float, default=-30.0)
    parser.add_argument("--diff-sll-db", type=float, default=-20.0)
    parser.add_argument("--null-depth-db", type=float, default=-35.0)
    parser.add_argument("--nulls", type=str, default="auto4")
    parser.add_argument("--sum-guard-deg", type=float, default=12.0)
    parser.add_argument("--diff-guard-deg", type=float, default=18.0)
    parser.add_argument("--diff-lobe-offset-deg", type=float, default=4.0)
    parser.add_argument("--fault-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--sll-topk", type=int, default=32)
    parser.add_argument("--w-sum-point", type=float, default=1.0)
    parser.add_argument("--w-sum-sll", type=float, default=0.45)
    parser.add_argument("--w-null", type=float, default=0.50)
    parser.add_argument("--w-diff-null", type=float, default=0.40)
    parser.add_argument("--w-diff-lobes", type=float, default=0.15)
    parser.add_argument("--w-diff-sll", type=float, default=1.00)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs_competition")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = args.sweep if args.sweep else [args.theta0]
    print(f"Using torch {torch.__version__}; cuda_available={torch.cuda.is_available()}")
    print(f"Array={args.rows}x{args.cols}, elements={args.rows * args.cols}, nulls={args.nulls}, fault_rate={args.fault_rate}")
    rows = [train_case(args, theta0) for theta0 in targets]
    write_summary(Path(args.output_dir) / "competition_summary.csv", rows)
    print(f"Saved outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
