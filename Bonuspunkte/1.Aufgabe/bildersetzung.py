from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return points in the order: top-left, top-right, bottom-right, bottom-left."""
    pts = points.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def detect_billboard_corners(scene_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """Detect the green billboard area and return ordered corner points plus a debug mask."""
    hsv = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2HSV)

    # Green chroma key range for the provided billboard image.
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1000:
        return None, mask

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(np.float32)

    return order_corners(corners), mask


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Perspektivische Bildersetzung mit OpenCV")
    parser.add_argument("--scene", default=str(base_dir / "Plakatwand.jpg"), help="Pfad zum Szenenbild")
    parser.add_argument("--insert", default=str(base_dir / "Welcome.jpg"), help="Pfad zum einzusetzenden Bild")
    parser.add_argument(
        "--out-composite",
        default=str(base_dir / "output_ersetzt.jpg"),
        help="Pfad fuer das finale ersetzte Bild",
    )
    parser.add_argument(
        "--out-warped",
        default=str(base_dir / "output_verzerrt.jpg"),
        help="Pfad fuer das perspektivisch verzerrte Bild",
    )
    parser.add_argument(
        "--out-matrix",
        default=str(base_dir / "transformationsmatrix.txt"),
        help="Pfad fuer die Homographie-Matrix",
    )
    parser.add_argument(
        "--out-points",
        default=str(base_dir / "erkannte_eckpunkte.txt"),
        help="Pfad fuer erkannte Zielkoordinaten",
    )
    parser.add_argument(
        "--out-mask",
        default=str(base_dir / "debug_gruenmaske.png"),
        help="Pfad fuer die erkannte Gruen-Maske (Debug)",
    )
    args = parser.parse_args()

    scene = cv2.imread(args.scene)
    insert = cv2.imread(args.insert)

    if scene is None:
        raise FileNotFoundError(f"Szenenbild nicht gefunden oder nicht lesbar: {args.scene}")
    if insert is None:
        raise FileNotFoundError(f"Insert-Bild nicht gefunden oder nicht lesbar: {args.insert}")

    dst_points, green_mask = detect_billboard_corners(scene)
    if dst_points is None:
        raise RuntimeError("Plakatflaeche konnte nicht automatisch erkannt werden.")

    insert_h, insert_w = insert.shape[:2]
    src_points = np.array(
        [[0, 0], [insert_w - 1, 0], [insert_w - 1, insert_h - 1], [0, insert_h - 1]],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(insert, homography, (scene.shape[1], scene.shape[0]))

    polygon_mask = np.zeros(scene.shape[:2], dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [dst_points.astype(np.int32)], 255)

    composite = scene.copy()
    composite[polygon_mask > 0] = warped[polygon_mask > 0]

    cv2.imwrite(args.out_warped, warped)
    cv2.imwrite(args.out_composite, composite)
    cv2.imwrite(args.out_mask, green_mask)

    np.savetxt(args.out_matrix, homography, fmt="%.8f")
    np.savetxt(args.out_points, dst_points, fmt="%.2f", header="x y")

    print("Erkannte Zielpunkte (tl, tr, br, bl):")
    print(dst_points)
    print("\nTransformationsmatrix H:")
    print(homography)
    print("\nGespeicherte Dateien:")
    print(args.out_composite)
    print(args.out_warped)
    print(args.out_matrix)
    print(args.out_points)
    print(args.out_mask)


if __name__ == "__main__":
    main()