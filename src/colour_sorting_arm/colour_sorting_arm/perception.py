"""Calibrated RGB tabletop edges, sparse wrist-IR plane fitting and CPU camera."""

from itertools import combinations
import numpy as np
import cv2
import mujoco

COLOURS = ("red", "green", "blue")


def table_frame(polygon):
    """Near corner of destination (smallest X, largest Y), +X outward, -Y across."""
    corners = np.asarray(polygon, float)[:, :2]
    origin_index = np.argmin(corners[:, 0] - corners[:, 1])
    origin = corners[origin_index]
    neighbours = corners[[(origin_index - 1) % 4, (origin_index + 1) % 4]]
    x_axis = neighbours[np.argmax(neighbours[:, 0])] - origin
    x_axis /= np.linalg.norm(x_axis)
    axes = np.array([x_axis, [x_axis[1], -x_axis[0]]])
    return origin, axes


def pixel_rays(pixels, calibration):
    K = np.asarray(calibration["K"], float)
    p = np.asarray(pixels, float).reshape(-1, 2)
    local = np.column_stack(
        [
            (p[:, 0] - K[0, 2]) / K[0, 0],
            -(p[:, 1] - K[1, 2]) / K[1, 1],
            -np.ones(len(p)),
        ]
    )
    return local @ np.asarray(calibration["rotation"], float).reshape(3, 3).T


def intersect_plane(pixels, calibration, normal, offset):
    rays = pixel_rays(pixels, calibration)
    origin = np.asarray(calibration["position"], float)
    normal = np.asarray(normal, float)
    denom = rays @ normal
    if np.any(abs(denom) < 1e-6):
        raise ValueError("Camera rays are parallel to the measured plane")
    distance = -(origin @ normal + offset) / denom
    if np.any(distance <= 0):
        raise ValueError("Measured plane is behind the camera")
    return origin + distance[:, None] * rays


def tabletop_mask(rgb):
    """Warm surfaces plus bright neutral highlights; retain dark-background contrast."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    warm = cv2.inRange(
        hsv, np.array([12, 8, 65], np.uint8), np.array([48, 145, 255], np.uint8)
    )
    neutral = cv2.inRange(
        hsv, np.array([0, 0, 180], np.uint8), np.array([179, 40, 255], np.uint8)
    )
    mask = cv2.bitwise_or(warm, neutral)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def camera_observation(rgb, position, rotation, fovy, ir_hit=None, stamp=0.0):
    """Only image pixels, camera calibration and a single physical range hit enter."""
    h, w = rgb.shape[:2]
    f = h / (2 * np.tan(np.deg2rad(fovy) / 2))
    calibration = dict(
        position=np.asarray(position).tolist(),
        rotation=np.asarray(rotation).reshape(3, 3).tolist(),
        K=[[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]],
        width=w,
        height=h,
    )
    mask = tabletop_mask(rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for contour in contours:
        if cv2.contourArea(contour) < 0.025 * w * h:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, 0.018 * perimeter, True)
        if len(poly) != 4:
            poly = cv2.approxPolyDP(hull, 0.025 * perimeter, True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        pts = poly[:, 0, :].astype(float)
        if (
            np.any(pts[:, 0] < 2)
            or np.any(pts[:, 0] > w - 3)
            or np.any(pts[:, 1] < 2)
            or np.any(pts[:, 1] > h - 3)
        ):
            continue
        if cv2.contourArea(contour) / max(cv2.contourArea(poly), 1) < 0.80:
            continue
        quads.append(pts.tolist())
    quads.sort(key=lambda q: float(np.mean(np.array(q)[:, 1])))
    table_roi = np.zeros((h, w), np.uint8)
    for quad in quads:
        cv2.fillConvexPoly(table_roi, np.asarray(quad, np.int32), 255)
    table_roi = cv2.erode(table_roi, np.ones((3, 3), np.uint8))
    blocks = {}
    image = rgb.astype(float)
    for i, colour in enumerate(COLOURS):
        cm = (
            (image[:, :, i] > 75)
            & (
                image[:, :, i]
                > 1.7 * np.max(image[:, :, [j for j in range(3) if j != i]], axis=2)
            )
            & (table_roi > 0)
        ).astype(np.uint8)
        cs, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        mom = cv2.moments(c)
        if mom["m00"] < 8:
            continue
        blocks[colour] = [mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]]
    return dict(
        stamp=float(stamp),
        calibration=calibration,
        table_pixels=quads,
        block_pixels=blocks,
        ir_hit=None if ir_hit is None else np.asarray(ir_hit).tolist(),
    )


def fit_plane(hits):
    points = np.asarray(hits, float)
    if points.shape[0] < 4:
        raise ValueError("Four spaced IR positions are required per tabletop")
    center = points.mean(0)
    _, singular, vh = np.linalg.svd(points - center)
    if singular[1] < 0.025:
        raise ValueError("IR samples are not sufficiently spread across the table")
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    offset = -normal @ center
    residual = float(np.max(abs(points @ normal + offset)))
    if normal[2] < np.cos(np.deg2rad(8)):
        raise ValueError(
            "IR plane tilt exceeds the supported 8 degrees; possible block/edge hit"
        )
    if residual > 0.004:
        raise ValueError("IR plane inconsistent: possible block hit or moving sensor")
    return normal, float(offset), residual


class EdgeMap:
    """Two image outlines plus four measured IR positions per table. No grid scan."""

    def __init__(self, source_center=None):
        self.source_center = (
            None if source_center is None else np.asarray(source_center).copy()
        )
        self.packet = None
        self.hits = {0: [], 1: []}
        self.tables = []
        self.probe_count = 0

    def begin(self, packet):
        if not packet or len(packet.get("table_pixels", [])) != 2:
            raise ValueError(
                "Need two complete tabletop outlines in the overview image"
            )
        self.packet = packet
        self.hits = {0: [], 1: []}
        self.probe_count = 0

    def candidates(self, table):
        """A planning-height prior picks safe probe locations; IR determines final geometry."""
        quad = np.asarray(self.packet["table_pixels"][table], float)
        world = intersect_plane(quad, self.packet["calibration"], [0, 0, 1], -0.24)
        blocked = []
        for pixel in self.packet["block_pixels"].values():
            blocked.append(
                intersect_plane([pixel], self.packet["calibration"], [0, 0, 1], -0.29)[
                    0, :2
                ]
            )
        points = []
        for u in (0.22, 0.5, 0.78):
            for v in (0.22, 0.5, 0.78):
                p = (1 - v) * ((1 - u) * world[0] + u * world[1]) + v * (
                    (1 - u) * world[3] + u * world[2]
                )
                if any(np.linalg.norm(p[:2] - b) < 0.055 for b in blocked):
                    continue
                points.append(np.array([p[0], p[1], 0.52]))
        return points

    @staticmethod
    def select_spread(candidates):
        """Candidates are (tool XYZ, IK joint target); choose four with maximum XY hull."""
        if len(candidates) < 4:
            raise ValueError("Not enough reachable, unoccupied IR probe positions")
        choices = list(combinations(candidates, 4))
        selected = max(
            choices,
            key=lambda group: cv2.contourArea(
                cv2.convexHull(np.array([x[0][:2] for x in group], np.float32))
            ),
        )
        if (
            cv2.contourArea(
                cv2.convexHull(np.array([x[0][:2] for x in selected], np.float32))
            )
            < 0.006
        ):
            raise ValueError("Reachable IR positions do not span enough tabletop area")
        return list(selected)

    def add_probe(self, table, hits):
        hits = np.asarray(hits, float)
        if hits.shape[0] < 5 or not np.isfinite(hits).all():
            raise ValueError("Missing valid IR measurements at a probe")
        point = np.median(hits, axis=0)
        if not 0.08 < point[2] < 0.4:
            raise ValueError(
                "IR did not hit a tabletop within the supported height range"
            )
        self.hits[table].append(point)
        self.probe_count += 1

    def finish(self):
        if self.packet is None:
            raise ValueError("No calibrated overview image")
        calibration = self.packet["calibration"]
        tables = []
        for i, quad in enumerate(self.packet["table_pixels"]):
            n, d, residual = fit_plane(self.hits[i])
            polygon = intersect_plane(quad, calibration, n, d)
            tables.append(
                dict(
                    polygon=polygon,
                    normal=n,
                    offset=d,
                    low=polygon[:, :2].min(0),
                    high=polygon[:, :2].max(0),
                    area_m2=float(
                        cv2.contourArea(polygon[:, :2].astype(np.float32)) / n[2]
                    ),
                    plane_residual_m=residual,
                )
            )
        all_objects = {}
        ownership = {}
        for colour, pixel in self.packet["block_pixels"].items():
            for i, t in enumerate(tables):
                top = intersect_plane(
                    [pixel],
                    calibration,
                    t["normal"],
                    t["offset"] - 0.05 * t["normal"][2],
                )[0]
                if (
                    cv2.pointPolygonTest(
                        t["polygon"][:, :2].astype(np.float32), tuple(top[:2]), True
                    )
                    > 0.012
                ):
                    all_objects[colour] = top - [0, 0, 0.025]
                    ownership[colour] = i
                    break
        if self.source_center is None:
            counts = [sum(v == i for v in ownership.values()) for i in range(2)]
            if max(counts) == 0:
                raise ValueError(
                    "No coloured blocks detected; cannot identify source table"
                )
            source = int(np.argmax(counts))
        else:
            source = int(
                np.argmin(
                    [
                        np.linalg.norm(t["polygon"][:, :2].mean(0) - self.source_center)
                        for t in tables
                    ]
                )
            )
        self.source_center = tables[source]["polygon"][:, :2].mean(0)
        destination = tables[1 - source]
        slots = []
        for x in np.arange(
            destination["low"][0] + 0.045, destination["high"][0] - 0.04, 0.075
        ):
            for y in np.arange(
                destination["low"][1] + 0.045, destination["high"][1] - 0.04, 0.075
            ):
                if (
                    cv2.pointPolygonTest(
                        destination["polygon"][:, :2].astype(np.float32),
                        (float(x), float(y)),
                        True,
                    )
                    < 0.043
                ):
                    continue
                if any(
                    ownership[c] == 1 - source
                    and np.linalg.norm(np.array([x, y]) - p[:2]) < 0.075
                    for c, p in all_objects.items()
                ):
                    continue
                n = destination["normal"]
                z = -(n[0] * x + n[1] * y + destination["offset"]) / n[2] + 0.025
                slots.append(np.array([x, y, z]))
        if not slots:
            raise ValueError(
                "No unoccupied placement cells inside the measured tabletop polygon"
            )
        self.destination = destination
        self.occupied = [
            p[:2].copy() for c, p in all_objects.items() if ownership[c] == 1 - source
        ]
        self.tables = tables
        return (
            {c: p for c, p in all_objects.items() if ownership[c] == source},
            slots,
            tables,
        )

    def serialized(self):
        return dict(
            method="camera_edges_sparse_ir",
            probe_positions=self.probe_count,
            roles=[
                "destination" if t is self.destination else "source"
                for t in self.tables
            ],
            tables=[
                {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in t.items()
                }
                for t in self.tables
            ],
        )


class SoftwareCamera:
    """Idealized ray-colour renderer for headless checks; no object-name masks."""

    def __init__(self, model, height=240, width=320, camera="overhead"):
        self.model = model
        self.height = height
        self.width = width
        self.cam = model.camera(camera).id
        f = height / (2 * np.tan(np.deg2rad(model.cam_fovy[self.cam]) / 2))
        y, x = np.indices((height, width))
        self.rays = np.stack(
            [(x - (width - 1) / 2) / f, -(y - (height - 1) / 2) / f, -np.ones_like(x)],
            axis=-1,
        ).reshape(-1, 3)
        self.ids = np.empty(height * width, np.int32)
        self.distance = np.empty(height * width)

    def render(self, data):
        rays = np.ascontiguousarray(
            (self.rays @ data.cam_xmat[self.cam].reshape(3, 3).T).ravel()
        )
        # Scalar ray intersections agree with MuJoCo's sensor geometry. On 3.3.7,
        # multiRay culling incorrectly omitted visible parts of these static boxes.
        for i, ray in enumerate(rays.reshape(-1, 3)):
            self.distance[i] = mujoco.mj_ray(
                self.model,
                data,
                data.cam_xpos[self.cam],
                ray,
                None,
                1,
                -1,
                self.ids[i : i + 1],
            )
        rgb = np.zeros((len(self.ids), 3), np.uint8)
        hit = self.ids >= 0
        rgba = self.model.geom_rgba[self.ids[hit], :3].copy()
        material = self.model.geom_matid[self.ids[hit]]
        has = material >= 0
        if np.any(has):
            rgba[has] *= self.model.mat_rgba[material[has], :3]
        rgb[hit] = np.clip(rgba * 255, 0, 255).astype(np.uint8)
        return rgb.reshape(self.height, self.width, 3)

    def close(self):
        pass
