#!/usr/bin/env python3
"""Interactive viewer for PAAL point clouds and meshes.

Open either:
  - a depthmap frame directory containing `pig_point_cloud.ply` and/or `pig_surface.obj`
  - a `.ply` point cloud file
  - a `.obj` mesh file

Examples:
    python view_pointcloud.py depthmap\\20260211-09-17-32\\pig0_20260211-09-17-49
    python view_pointcloud.py depthmap\\20260211-09-17-32\\pig0_20260211-09-17-49\\pig_point_cloud.ply
    python view_pointcloud.py depthmap\\20260211-09-17-32\\pig0_20260211-09-17-49 --mode both
    python view_pointcloud.py depthmap\\20260211-09-17-32\\pig0_20260211-09-17-49 --save outputs\\pig0_view.png --no-show
"""

import argparse
import os
from pathlib import Path

import numpy as np


def resolve_inputs(path_str: str, mode: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    ply_path = None
    obj_path = None

    if path.is_dir():
        ply_candidate = path / "pig_point_cloud.ply"
        obj_candidate = path / "pig_surface.obj"
        if ply_candidate.exists():
            ply_path = ply_candidate
        if obj_candidate.exists():
            obj_path = obj_candidate
    elif path.suffix.lower() == ".ply":
        ply_path = path
        obj_candidate = path.with_name("pig_surface.obj")
        if obj_candidate.exists():
            obj_path = obj_candidate
    elif path.suffix.lower() == ".obj":
        obj_path = path
        ply_candidate = path.with_name("pig_point_cloud.ply")
        if ply_candidate.exists():
            ply_path = ply_candidate
    else:
        raise ValueError(f"Unsupported input: {path}")

    if mode == "pointcloud" and ply_path is None:
        raise FileNotFoundError("Point cloud file not found (`pig_point_cloud.ply`)")
    if mode == "mesh" and obj_path is None:
        raise FileNotFoundError("Mesh file not found (`pig_surface.obj`)")
    if mode == "both" and (ply_path is None or obj_path is None):
        missing = []
        if ply_path is None:
            missing.append("pig_point_cloud.ply")
        if obj_path is None:
            missing.append("pig_surface.obj")
        raise FileNotFoundError(f"Missing required file(s) for --mode both: {', '.join(missing)}")

    if mode == "auto":
        if ply_path is not None:
            mode = "pointcloud"
        elif obj_path is not None:
            mode = "mesh"
        else:
            raise FileNotFoundError("No viewer-compatible files found")

    return ply_path, obj_path, mode


def infer_frame_dir(input_path: Path, ply_path: Path | None, obj_path: Path | None):
    if input_path.is_dir():
        return input_path

    for candidate in (ply_path, obj_path, input_path):
        if candidate is None:
            continue
        parent = candidate.parent
        if (parent / "depthmap_bundle.npz").exists():
            return parent

    return None


def find_ir_image(frame_dir: Path | None):
    if frame_dir is None:
        return None

    parts = frame_dir.name.split("_", 1)
    if len(parts) != 2:
        fallback = frame_dir / "mesh_texture.png"
        return fallback if fallback.exists() else None

    pig_id, capture_ts = parts
    session_ts = frame_dir.parent.name
    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / "images" / session_ts / f"{pig_id}_ir_vis_{capture_ts}.jpg",
        repo_root / "images" / session_ts / f"{pig_id}_ir_vis_{capture_ts}_cropped.jpg",
        frame_dir / "mesh_texture.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_ascii_ply(path: Path):
    vertex_count = None
    properties = []
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
        if line != "ply":
            raise ValueError(f"Unsupported PLY file: {path}")

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"PLY vertex count missing: {path}")

        data = np.loadtxt(f, dtype=np.float64, max_rows=vertex_count)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    prop_idx = {name: idx for idx, name in enumerate(properties)}
    xyz = np.stack(
        [
            data[:, prop_idx["x"]],
            data[:, prop_idx["y"]],
            data[:, prop_idx["z"]],
        ],
        axis=1,
    ).astype(np.float32)

    colors = None
    if all(name in prop_idx for name in ("red", "green", "blue")):
        colors = np.stack(
            [
                data[:, prop_idx["red"]],
                data[:, prop_idx["green"]],
                data[:, prop_idx["blue"]],
            ],
            axis=1,
        ).astype(np.float32) / 255.0

    return xyz, colors


def load_obj_mesh(path: Path):
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, xs, ys, zs = line.split()[:4]
                vertices.append((float(xs), float(ys), float(zs)))
            elif line.startswith("f "):
                parts = line.split()[1:]
                indices = []
                for part in parts:
                    idx = part.split("/")[0]
                    if idx:
                        indices.append(int(idx) - 1)
                if len(indices) < 3:
                    continue
                for i in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[i], indices[i + 1]))

    if not vertices or not faces:
        raise ValueError(f"OBJ mesh is empty or unsupported: {path}")

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def subsample_points(vertices: np.ndarray, colors: np.ndarray | None, max_points: int):
    if len(vertices) <= max_points:
        return vertices, colors

    idx = np.linspace(0, len(vertices) - 1, max_points, dtype=np.int32)
    if colors is None:
        return vertices[idx], None
    return vertices[idx], colors[idx]


def set_axes_equal(ax, xyz: np.ndarray):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def display_vertices(vertices: np.ndarray, flip_y: bool):
    shown = vertices.copy()
    if flip_y:
        shown[:, 1] *= -1.0
    return shown


def build_title(args, ply_path: Path | None, obj_path: Path | None, mode: str):
    if args.title:
        return args.title
    source = ply_path or obj_path
    if source is None:
        return "Point Cloud Viewer"
    return f"{source.parent.name} ({mode})"


def rotation_matrix(yaw_deg: float, pitch_deg: float):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    rot_y = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float32,
    )
    return rot_x @ rot_y


def hex_color(rgb):
    r, g, b = np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def run_tk_viewer(args, ply_path: Path | None, obj_path: Path | None, mode: str):
    import tkinter as tk

    if args.save:
        raise RuntimeError("`--save` requires matplotlib; it is not installed in this Python environment.")
    if args.no_show:
        raise RuntimeError("`--no-show` requires `--save`. Interactive fallback uses a Tk window.")

    points = None
    colors = None
    faces = None
    mesh_points = None
    ir_canvas = None
    ir_state = {"image": None, "photo": None, "path": None}

    if mode in {"pointcloud", "both"} and ply_path is not None:
        points, colors = load_ascii_ply(ply_path)
        tk_limit = min(max(1, args.max_points), 12000)
        points, colors = subsample_points(points, colors, tk_limit)
        print(f"Loaded point cloud: {ply_path}")
        print(f"Displayed points: {len(points):,} (Tk fallback)")

    if mode in {"mesh", "both"} and obj_path is not None:
        mesh_vertices, mesh_faces = load_obj_mesh(obj_path)
        if points is None:
            points = mesh_vertices
        mesh_points = mesh_vertices
        faces = mesh_faces
        print(f"Loaded mesh: {obj_path}")
        print(f"Vertices: {len(mesh_vertices):,} | Faces: {len(mesh_faces):,}")

    if points is None:
        raise RuntimeError("Nothing to display")

    points = display_vertices(points, flip_y=not args.no_flip_y)
    center = points.mean(axis=0, keepdims=True)
    points_centered = points - center
    radius = max(float(np.linalg.norm(points_centered, axis=1).max()), 1e-6)

    root = tk.Tk()
    root.title(build_title(args, ply_path, obj_path, mode))

    input_path = Path(args.path)
    frame_dir = infer_frame_dir(input_path, ply_path, obj_path)
    ir_path = find_ir_image(frame_dir)
    if ir_path is not None:
        try:
            from PIL import Image, ImageTk

            resampling = getattr(Image, "Resampling", Image)
            with Image.open(ir_path) as img:
                ir_state["image"] = img.convert("L").convert("RGB")
            ir_state["path"] = ir_path
            ir_state["resampling"] = resampling.LANCZOS
            print(f"Loaded IR preview: {ir_path}")
        except Exception as exc:
            print(f"IR preview unavailable: {exc}")

    root.configure(bg="#111111")
    root.geometry("1560x860" if ir_state["image"] is not None else "1100x820")

    main_frame = tk.Frame(root, bg="#111111")
    main_frame.pack(fill="both", expand=True)

    if ir_state["image"] is not None:
        left_frame = tk.Frame(main_frame, bg="#111111", width=480)
        left_frame.pack(side="left", fill="y")

        ir_header = tk.Label(
            left_frame,
            text=f"IR image: {ir_state['path'].name}",
            anchor="w",
            justify="left",
            padx=12,
            pady=10,
            fg="white",
            bg="#111111",
            font=("Segoe UI", 11, "bold"),
        )
        ir_header.pack(fill="x")

        ir_subheader = tk.Label(
            left_frame,
            text="This is the grayscale IR frame used to color the point cloud / mesh.",
            anchor="w",
            justify="left",
            padx=12,
            pady=0,
            fg="#cfcfcf",
            bg="#111111",
            font=("Segoe UI", 10),
        )
        ir_subheader.pack(fill="x")

        ir_canvas = tk.Canvas(
            left_frame,
            width=460,
            height=760,
            bg="black",
            highlightthickness=1,
            highlightbackground="#444444",
        )
        ir_canvas.pack(fill="both", expand=True, padx=12, pady=(10, 12))

    canvas = tk.Canvas(main_frame, width=1080, height=820, bg="black", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    state = {
        "yaw": args.azim,
        "pitch": args.elev,
        "zoom": 1.0,
        "pan_x": 0.0,
        "pan_y": 0.0,
        "drag_mode": None,
        "last_x": 0,
        "last_y": 0,
        "initialized": False,
    }

    def normalized_projection(verts, yaw_deg=None, pitch_deg=None):
        yaw = state["yaw"] if yaw_deg is None else yaw_deg
        pitch = state["pitch"] if pitch_deg is None else pitch_deg
        rot = rotation_matrix(yaw, pitch)
        rotated = verts @ rot.T
        depth = rotated[:, 2]
        z_offset = depth - depth.min() + 2.5 * radius
        x2 = rotated[:, 0] / z_offset
        y2 = rotated[:, 1] / z_offset
        return x2, y2, depth

    def fit_zoom():
        w = max(canvas.winfo_width(), 1)
        h = max(canvas.winfo_height(), 1)
        scale = max(min(w, h), 1)
        x2, y2, _depth = normalized_projection(points_centered)
        finite = np.isfinite(x2) & np.isfinite(y2)
        if not np.any(finite):
            return 1.0

        x_abs = np.abs(x2[finite])
        y_abs = np.abs(y2[finite])
        x_span = max(float(np.quantile(x_abs, 0.995)), 1e-6)
        y_span = max(float(np.quantile(y_abs, 0.995)), 1e-6)

        zoom_x = (0.46 * w) / (scale * x_span)
        zoom_y = (0.46 * h) / (scale * y_span)
        return float(np.clip(min(zoom_x, zoom_y), 0.05, 50.0))

    def project(verts):
        x2, y2, depth = normalized_projection(verts)

        w = max(canvas.winfo_width(), 1)
        h = max(canvas.winfo_height(), 1)
        scale = max(min(w, h), 1)
        sx = x2 * state["zoom"] * scale + (w / 2.0) + state["pan_x"]
        sy = -y2 * state["zoom"] * scale + (h / 2.0) + state["pan_y"]
        return sx, sy, depth

    def redraw_ir():
        if ir_canvas is None or ir_state["image"] is None:
            return

        w = max(ir_canvas.winfo_width(), 1)
        h = max(ir_canvas.winfo_height(), 1)
        image = ir_state["image"].copy()
        image.thumbnail((max(w - 18, 1), max(h - 18, 1)), ir_state["resampling"])

        ir_state["photo"] = ImageTk.PhotoImage(image)
        ir_canvas.delete("all")
        ir_canvas.create_image(w / 2.0, h / 2.0, image=ir_state["photo"], anchor="center")
        ir_canvas.create_text(
            10,
            h - 10,
            anchor="sw",
            fill="#e8e8e8",
            text="IR grayscale",
            font=("Segoe UI", 11, "bold"),
        )

    def redraw():
        if not state["initialized"]:
            state["zoom"] = fit_zoom()
            state["initialized"] = True

        canvas.delete("all")
        sx, sy, depth = project(points_centered)
        order = np.argsort(depth)

        if colors is None:
            point_colors = np.tile(np.array([[0.85, 0.85, 0.85]], dtype=np.float32), (len(points_centered), 1))
        else:
            point_colors = colors

        radius_px = max(1, int(round(args.point_size)))
        for idx in order:
            x = sx[idx]
            y = sy[idx]
            if x < -5 or y < -5 or x > canvas.winfo_width() + 5 or y > canvas.winfo_height() + 5:
                continue
            color = hex_color(point_colors[idx])
            if radius_px <= 1:
                canvas.create_line(x, y, x + 1, y, fill=color)
            else:
                canvas.create_oval(x - radius_px, y - radius_px, x + radius_px, y + radius_px,
                                   outline="", fill=color)

        if faces is not None:
            shown_mesh = display_vertices(mesh_points, flip_y=not args.no_flip_y) - center
            mx, my, mz = project(shown_mesh)
            for face in faces[:15000]:
                pts = np.array([[mx[face[0]], my[face[0]]], [mx[face[1]], my[face[1]]], [mx[face[2]], my[face[2]]]])
                if np.any(~np.isfinite(pts)):
                    continue
                shade = int(np.clip(160 + 60 * np.mean(mz[list(face)]) / max(radius, 1e-6), 80, 220))
                edge = f"#{shade:02x}{shade:02x}{shade:02x}"
                canvas.create_line(*pts[0], *pts[1], fill=edge)
                canvas.create_line(*pts[1], *pts[2], fill=edge)
                canvas.create_line(*pts[2], *pts[0], fill=edge)

        msg = "Left drag=rotate | Right drag=pan | Wheel=zoom | F/R=fit/reset | Q/Esc=quit"
        canvas.create_text(12, 12, anchor="nw", fill="white", text=msg, font=("Segoe UI", 11))
        redraw_ir()

    def start_drag(event, mode_name):
        state["drag_mode"] = mode_name
        state["last_x"] = event.x
        state["last_y"] = event.y

    def on_drag(event):
        dx = event.x - state["last_x"]
        dy = event.y - state["last_y"]
        state["last_x"] = event.x
        state["last_y"] = event.y
        if state["drag_mode"] == "rotate":
            state["yaw"] += dx * 0.5
            state["pitch"] += dy * 0.5
        elif state["drag_mode"] == "pan":
            state["pan_x"] += dx
            state["pan_y"] += dy
        redraw()

    def end_drag(_event):
        state["drag_mode"] = None

    def on_wheel(event):
        delta = getattr(event, "delta", 0)
        if delta == 0 and hasattr(event, "num"):
            delta = 120 if event.num == 4 else -120
        factor = 1.1 if delta > 0 else 1 / 1.1
        state["zoom"] *= factor
        redraw()

    def on_reset(_event=None):
        state["yaw"] = args.azim
        state["pitch"] = args.elev
        state["zoom"] = fit_zoom()
        state["pan_x"] = 0.0
        state["pan_y"] = 0.0
        redraw()

    def on_fit(_event=None):
        state["zoom"] = fit_zoom()
        state["pan_x"] = 0.0
        state["pan_y"] = 0.0
        redraw()

    canvas.bind("<ButtonPress-1>", lambda e: start_drag(e, "rotate"))
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", end_drag)
    canvas.bind("<ButtonPress-3>", lambda e: start_drag(e, "pan"))
    canvas.bind("<B3-Motion>", on_drag)
    canvas.bind("<ButtonRelease-3>", end_drag)
    canvas.bind("<MouseWheel>", on_wheel)
    canvas.bind("<Button-4>", on_wheel)
    canvas.bind("<Button-5>", on_wheel)
    root.bind("r", on_reset)
    root.bind("R", on_reset)
    root.bind("f", on_fit)
    root.bind("F", on_fit)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.bind("q", lambda _e: root.destroy())
    root.bind("Q", lambda _e: root.destroy())
    canvas.bind("<Configure>", lambda _e: redraw())
    if ir_canvas is not None:
        ir_canvas.bind("<Configure>", lambda _e: redraw_ir())

    print("Controls: left-drag=rotate, right-drag=pan, scroll=zoom, f=fit, r=reset, q=quit")
    redraw()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="View a PAAL point cloud or mesh in 3D")
    parser.add_argument("path", help="Depthmap frame directory or a .ply/.obj file")
    parser.add_argument("--mode", choices=["auto", "pointcloud", "mesh", "both"], default="auto",
                        help="What to render (default: auto)")
    parser.add_argument("--max-points", type=int, default=30000,
                        help="Maximum number of points to draw for the point cloud")
    parser.add_argument("--point-size", type=float, default=1.2,
                        help="Marker size for point-cloud rendering")
    parser.add_argument("--mesh-alpha", type=float, default=0.75,
                        help="Opacity for mesh rendering")
    parser.add_argument("--elev", type=float, default=20.0, help="Initial elevation angle")
    parser.add_argument("--azim", type=float, default=-60.0, help="Initial azimuth angle")
    parser.add_argument("--title", default="", help="Optional plot title")
    parser.add_argument("--save", default="", help="Optional PNG path for a saved screenshot")
    parser.add_argument("--no-show", action="store_true", help="Save only, do not open an interactive window")
    parser.add_argument("--no-flip-y", action="store_true",
                        help="Do not flip Y for display (default keeps the pig upright)")
    args = parser.parse_args()

    input_path = Path(args.path)
    ply_path, obj_path, mode = resolve_inputs(args.path, args.mode)
    frame_dir = infer_frame_dir(input_path, ply_path, obj_path)
    ir_path = find_ir_image(frame_dir)

    try:
        import matplotlib
        if args.no_show and args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ModuleNotFoundError:
        run_tk_viewer(args, ply_path, obj_path, mode)
        return

    if ir_path is not None:
        fig = plt.figure(figsize=(14, 8))
        grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5])
        ax_ir = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[0, 1], projection="3d")
        try:
            from PIL import Image

            with Image.open(ir_path) as img:
                ir_image = np.asarray(img.convert("L"))
            ax_ir.imshow(ir_image, cmap="gray")
            ax_ir.set_title("IR")
            ax_ir.axis("off")
            print(f"Loaded IR preview: {ir_path}")
        except Exception as exc:
            print(f"IR preview unavailable: {exc}")
            ax_ir.axis("off")
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    title = build_title(args, ply_path, obj_path, mode)
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

    all_xyz = []

    if mode in {"pointcloud", "both"}:
        vertices, colors = load_ascii_ply(ply_path)
        vertices, colors = subsample_points(vertices, colors, max(1, args.max_points))
        shown = display_vertices(vertices, flip_y=not args.no_flip_y)
        if colors is None:
            colors = cm.gray(np.linspace(0.2, 0.9, len(shown)))[:, :3]
        ax.scatter(shown[:, 0], shown[:, 1], shown[:, 2], c=colors, s=args.point_size, linewidths=0)
        all_xyz.append(shown)
        print(f"Loaded point cloud: {ply_path}")
        print(f"Displayed points: {len(shown):,}")

    if mode in {"mesh", "both"}:
        mesh_vertices, mesh_faces = load_obj_mesh(obj_path)
        shown_mesh = display_vertices(mesh_vertices, flip_y=not args.no_flip_y)
        tris = shown_mesh[mesh_faces]
        face_z = tris[:, :, 2].mean(axis=1)
        z_min = float(face_z.min())
        z_max = float(face_z.max())
        denom = z_max - z_min if z_max > z_min else 1.0
        face_colors = cm.viridis((face_z - z_min) / denom)
        face_colors[:, 3] = args.mesh_alpha
        poly = Poly3DCollection(tris, facecolors=face_colors, edgecolors="none", linewidths=0.0)
        ax.add_collection3d(poly)
        all_xyz.append(shown_mesh)
        print(f"Loaded mesh: {obj_path}")
        print(f"Vertices: {len(mesh_vertices):,} | Faces: {len(mesh_faces):,}")

    if not all_xyz:
        raise RuntimeError("Nothing to display")

    xyz = np.vstack(all_xyz)
    set_axes_equal(ax, xyz)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(False)

    print("Controls: left-drag=rotate, right-drag=pan, scroll=zoom")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved screenshot: {save_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
