import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn


class ArrayNet(nn.Module):
    """MLP that maps beam requirements to per-element amplitude and phase."""

    def __init__(self, element_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * element_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_planar_positions(rows: int, cols: int, spacing: float, device: torch.device) -> torch.Tensor:
    xs = (torch.arange(cols, dtype=torch.float32, device=device) - (cols - 1) / 2.0) * spacing
    ys = (torch.arange(rows, dtype=torch.float32, device=device) - (rows - 1) / 2.0) * spacing
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def network_output_to_weights(output: torch.Tensor, element_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    amp_raw = output[:, :element_count]
    phase_raw = output[:, element_count:]
    amplitude = torch.sigmoid(amp_raw)
    phase = math.pi * torch.tanh(phase_raw)
    weights = amplitude * torch.exp(1j * phase)
    return weights, amplitude, phase


def array_factor_1d(weights: torch.Tensor, positions: torch.Tensor, theta_rad: torch.Tensor, phi_rad: float) -> torch.Tensor:
    k = 2.0 * math.pi
    u = torch.sin(theta_rad) * math.cos(phi_rad)
    v = torch.sin(theta_rad) * math.sin(phi_rad)
    phase = k * (positions[:, 0:1] * u[None, :] + positions[:, 1:2] * v[None, :])
    steering = torch.exp(1j * phase)
    af = torch.sum(weights[:, :, None] * steering[None, :, :], dim=1)
    return torch.abs(af)


def normalized_db(af: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    af_norm = af / (torch.amax(af, dim=1, keepdim=True) + eps)
    return 20.0 * torch.log10(torch.clamp(af_norm, min=eps))


def compute_loss(
    pattern_db: torch.Tensor,
    amplitude: torch.Tensor,
    theta_deg: torch.Tensor,
    theta0_deg: float,
    target_sll_db: float,
    mainlobe_guard_deg: float,
    sll_loss_weight: float,
    amp_loss_weight: float,
    sll_topk: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_index = int(torch.argmin(torch.abs(theta_deg - theta0_deg)).item())
    target_gain_db = pattern_db[:, target_index]

    main_region = torch.abs(theta_deg - theta0_deg) <= mainlobe_guard_deg
    side_region = ~main_region
    side_pattern = pattern_db[:, side_region]

    loss_pointing = torch.mean((-target_gain_db) ** 2)
    excess_sll = torch.relu(side_pattern - target_sll_db)
    topk = min(sll_topk, excess_sll.shape[1])
    loss_sll = torch.mean(torch.topk(excess_sll, k=topk, dim=1).values ** 2)
    loss_amp = torch.mean((amplitude - 0.70) ** 2)
    loss = 1.0 * loss_pointing + sll_loss_weight * loss_sll + amp_loss_weight * loss_amp

    metrics = {
        "loss_pointing": float(loss_pointing.detach().cpu()),
        "loss_sll": float(loss_sll.detach().cpu()),
        "loss_amp": float(loss_amp.detach().cpu()),
    }
    return loss, metrics


def evaluate_pattern(theta_deg: torch.Tensor, pattern_db: torch.Tensor, theta0_deg: float, guard_deg: float) -> dict[str, float]:
    values = pattern_db[0].detach().cpu()
    angles = theta_deg.detach().cpu()
    peak_idx = int(torch.argmax(values).item())
    main_region = torch.abs(angles - theta0_deg) <= guard_deg
    side_region = ~main_region
    sll = float(torch.amax(values[side_region]).item())
    return {
        "peak_theta_deg": float(angles[peak_idx].item()),
        "pointing_error_deg": abs(float(angles[peak_idx].item()) - theta0_deg),
        "target_gain_db": float(values[int(torch.argmin(torch.abs(angles - theta0_deg)).item())].item()),
        "sll_db": sll,
    }


def save_pattern_csv(path: Path, theta_deg: torch.Tensor, pattern_db: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_deg", "pattern_db"])
        for theta, value in zip(theta_deg.detach().cpu().tolist(), pattern_db[0].detach().cpu().tolist()):
            writer.writerow([f"{theta:.4f}", f"{value:.6f}"])


def save_weights_csv(path: Path, amplitude: torch.Tensor, phase: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    amp = amplitude[0].detach().cpu().tolist()
    ph = phase[0].detach().cpu().tolist()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["element", "amplitude", "phase_rad", "phase_deg"])
        for idx, (a, p) in enumerate(zip(amp, ph)):
            writer.writerow([idx, f"{a:.8f}", f"{p:.8f}", f"{math.degrees(p):.4f}"])


def save_svg_plot(
    path: Path,
    theta_deg: torch.Tensor,
    pattern_db: torch.Tensor,
    theta0_deg: float,
    target_sll_db: float,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    angles = theta_deg.detach().cpu().tolist()
    values = [max(-60.0, min(2.0, v)) for v in pattern_db[0].detach().cpu().tolist()]
    width, height = 920, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 25, 50, 65
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x_min, x_max = -90.0, 90.0
    y_min, y_max = -60.0, 2.0

    def xmap(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def ymap(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    points = " ".join(f"{xmap(x):.2f},{ymap(y):.2f}" for x, y in zip(angles, values))
    grid_lines = []
    for x_tick in range(-90, 91, 30):
        x = xmap(float(x_tick))
        grid_lines.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#e5e7eb"/>')
        grid_lines.append(f'<text x="{x:.1f}" y="{height - 35}" text-anchor="middle" font-size="13" fill="#374151">{x_tick}</text>')
    for y_tick in range(-60, 1, 10):
        y = ymap(float(y_tick))
        grid_lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        grid_lines.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="13" fill="#374151">{y_tick}</text>')

    target_x = xmap(theta0_deg)
    sll_y = ymap(target_sll_db)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin_left}" y="30" font-size="20" font-family="Arial, sans-serif" fill="#111827">{title}</text>
  <g font-family="Arial, sans-serif">
    {''.join(grid_lines)}
    <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827" stroke-width="1.2"/>
    <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827" stroke-width="1.2"/>
    <line x1="{target_x:.2f}" y1="{margin_top}" x2="{target_x:.2f}" y2="{height - margin_bottom}" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="6 5"/>
    <line x1="{margin_left}" y1="{sll_y:.2f}" x2="{width - margin_right}" y2="{sll_y:.2f}" stroke="#0f766e" stroke-width="1.4" stroke-dasharray="7 5"/>
    <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2.2"/>
    <text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="15" fill="#111827">Theta (deg), phi=0 deg</text>
    <text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-size="15" fill="#111827">Normalized pattern (dB)</text>
    <text x="{target_x + 8:.1f}" y="{margin_top + 18}" font-size="13" fill="#ef4444">target {theta0_deg:g} deg</text>
    <text x="{width - margin_right - 135}" y="{sll_y - 8:.1f}" font-size="13" fill="#0f766e">SLL target {target_sll_db:g} dB</text>
  </g>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def train_one(args: argparse.Namespace, theta0_deg: float) -> dict[str, float]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows, cols = args.rows, args.cols
    element_count = rows * cols
    positions = build_planar_positions(rows, cols, args.spacing, device)
    theta_deg = torch.linspace(args.theta_min, args.theta_max, args.theta_points, device=device)
    theta_rad = torch.deg2rad(theta_deg)
    phi_rad = math.radians(args.phi0)

    model = ArrayNet(element_count).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x = torch.tensor(
        [[theta0_deg / 90.0, args.phi0 / 180.0, args.target_sll_db / 60.0]],
        dtype=torch.float32,
        device=device,
    )

    start = time.perf_counter()
    last_metrics = {}
    for epoch in range(1, args.epochs + 1):
        output = model(x)
        weights, amplitude, phase = network_output_to_weights(output, element_count)
        af = array_factor_1d(weights, positions, theta_rad, phi_rad)
        pattern_db = normalized_db(af)
        loss, last_metrics = compute_loss(
            pattern_db,
            amplitude,
            theta_deg,
            theta0_deg,
            args.target_sll_db,
            args.mainlobe_guard_deg,
            args.sll_loss_weight,
            args.amp_loss_weight,
            args.sll_topk,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            eval_metrics = evaluate_pattern(theta_deg, pattern_db, theta0_deg, args.mainlobe_guard_deg)
            print(
                f"theta0={theta0_deg:>6.1f} epoch={epoch:>5d} "
                f"loss={loss.item():.5f} peak={eval_metrics['peak_theta_deg']:.2f} "
                f"target={eval_metrics['target_gain_db']:.2f}dB sll={eval_metrics['sll_db']:.2f}dB"
            )

    elapsed = time.perf_counter() - start
    output = model(x)
    weights, amplitude, phase = network_output_to_weights(output, element_count)
    af = array_factor_1d(weights, positions, theta_rad, phi_rad)
    pattern_db = normalized_db(af)
    eval_metrics = evaluate_pattern(theta_deg, pattern_db, theta0_deg, args.mainlobe_guard_deg)
    eval_metrics.update(last_metrics)
    eval_metrics["theta0_deg"] = theta0_deg
    eval_metrics["train_seconds"] = elapsed
    eval_metrics["epochs"] = args.epochs
    eval_metrics["elements"] = element_count

    stem = f"theta_{theta0_deg:+.0f}".replace("+", "p").replace("-", "m")
    out_dir = Path(args.output_dir)
    save_pattern_csv(out_dir / f"pattern_{stem}.csv", theta_deg, pattern_db)
    save_weights_csv(out_dir / f"weights_{stem}.csv", amplitude, phase)
    save_svg_plot(
        out_dir / f"pattern_{stem}.svg",
        theta_deg,
        pattern_db,
        theta0_deg,
        args.target_sll_db,
        f"PI-NAS {rows}x{cols} beam synthesis, theta0={theta0_deg:g} deg",
    )
    return eval_metrics


def write_summary(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "theta0_deg",
        "peak_theta_deg",
        "pointing_error_deg",
        "target_gain_db",
        "sll_db",
        "train_seconds",
        "epochs",
        "elements",
        "loss_pointing",
        "loss_sll",
        "loss_amp",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PI-NAS PyTorch demo for AI antenna array synthesis.")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=0.5, help="Element spacing in wavelengths.")
    parser.add_argument("--theta0", type=float, default=30.0)
    parser.add_argument("--phi0", type=float, default=0.0)
    parser.add_argument("--target-sll-db", type=float, default=-20.0)
    parser.add_argument("--mainlobe-guard-deg", type=float, default=20.0)
    parser.add_argument("--sll-loss-weight", type=float, default=0.50)
    parser.add_argument("--amp-loss-weight", type=float, default=0.0)
    parser.add_argument("--sll-topk", type=int, default=24)
    parser.add_argument("--theta-min", type=float, default=-90.0)
    parser.add_argument("--theta-max", type=float, default=90.0)
    parser.add_argument("--theta-points", type=int, default=721)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--sweep", type=float, nargs="*", default=None, help="Run multiple target theta angles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    theta_targets = args.sweep if args.sweep else [args.theta0]
    print(f"Using torch {torch.__version__}; cuda_available={torch.cuda.is_available()}")
    results = [train_one(args, theta0) for theta0 in theta_targets]
    write_summary(Path(args.output_dir) / "summary.csv", results)
    print(f"Saved outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
