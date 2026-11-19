"""Calibrated image/IR map tests, including offset, tilt and unequal table heights."""

from pathlib import Path
import json
from validate_controller import run, ROOT


def main():
    cases = [
        dict(camera_offset=0.0, camera_tilt=0.0),
        dict(camera_offset=0.08, camera_tilt=2.0),
        dict(camera_offset=0.08, camera_tilt=-2.0, destination_height=0.26),
    ]
    results = []
    for case in cases:
        r = run(colours=(), mapping_only=True, **case)
        results.append(r)
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "physical_success",
                        "sim_time",
                        "mapping_scores",
                        "object_errors_mm",
                        "camera_offset",
                        "camera_tilt",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
    out = ROOT / "results/edge_mapping_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    assert all(
        r["physical_success"]
        and len(r["object_errors_mm"]) == 3
        and max(s["max_corner_error_mm"] for s in r["mapping_scores"]) < 15
        for r in results
    )


if __name__ == "__main__":
    main()
