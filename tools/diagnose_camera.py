"""Capture the exact calibrated camera input and detector rejection reasons.
Run with mjpython in a desktop terminal. No ROS nodes or motion tuning involved.
"""

from pathlib import Path
import sys, json, argparse
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/colour_sorting_arm"))
from colour_sorting_arm.simulation import Engine
from colour_sorting_arm.perception import (
    SoftwareCamera,
    camera_observation,
    tabletop_mask,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["opengl", "software"], default="opengl")
    args = parser.parse_args()
    engine = Engine(ROOT / "src/colour_sorting_arm/models/scene.xml")
    target = engine.data.ctrl.copy()
    for _ in range(round(1.2 / engine.model.opt.timestep)):
        engine.step(target)
    if args.backend == "software":
        renderer = SoftwareCamera(
            engine.model, height=480, width=640, camera="overhead"
        )
        rgb = renderer.render(engine.data)
    else:
        import mujoco

        renderer = mujoco.Renderer(engine.model, height=480, width=640)
        renderer.update_scene(engine.data, camera="overhead")
        rgb = renderer.render().copy()
    renderer.close()
    cid = engine.model.camera("overhead").id
    packet = camera_observation(
        rgb,
        engine.data.cam_xpos[cid],
        engine.data.cam_xmat[cid],
        engine.model.cam_fovy[cid],
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = tabletop_mask(rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    details = []
    h, w = rgb.shape[:2]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, 0.018 * perimeter, True)
        if len(poly) != 4:
            poly = cv2.approxPolyDP(hull, 0.025 * perimeter, True)
        pts = poly[:, 0, :]
        ratio = area / max(cv2.contourArea(poly), 1)
        reasons = []
        if area < 0.025 * w * h:
            reasons.append("area below 2.5% of image")
        if len(poly) != 4:
            reasons.append(f"{len(poly)} corners instead of four")
        if not cv2.isContourConvex(poly):
            reasons.append("non-convex polygon")
        if (
            np.any(pts[:, 0] < 2)
            or np.any(pts[:, 0] > w - 3)
            or np.any(pts[:, 1] < 2)
            or np.any(pts[:, 1] > h - 3)
        ):
            reasons.append("touches image border")
        if ratio < 0.80:
            reasons.append("contour/polygon fill below 80%")
        details.append(
            dict(
                area_pixels=float(area),
                corners=len(poly),
                fill_ratio=float(ratio),
                rejected_by=reasons,
            )
        )
    report = dict(
        backend=args.backend,
        accepted_outlines=len(packet["table_pixels"]),
        mask_fraction=float(np.mean(mask > 0)),
        bright_pixels_below_minimum_saturation=float(
            np.mean((hsv[:, :, 2] > 100) & (hsv[:, :, 1] < 8))
        ),
        contours=details,
        calibration=packet["calibration"],
    )
    out = ROOT / "results/camera_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(out / f"{args.backend}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(str(out / f"{args.backend}_mask.png"), mask)
    (out / f"{args.backend}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Saved diagnostic frames:", out)


if __name__ == "__main__":
    main()
