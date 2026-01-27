"""
STEP analyzer module for extracting parameters from existing STEP files.

Identifies profile vertices and reverse-engineers dimensions (L1-L10, R1, R2).
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math
import logging

from build123d import import_step, Shape, Vertex, Edge, Wire, GeomType
from .geometry import ProfileParams, EdgeInfo

logger = logging.getLogger(__name__)


def _sample_edge_points_yz(edge: Edge, num_points: int = 33) -> Optional[List[Tuple[float, float]]]:
    """
    Sample points along an edge for plotting in Y-Z.

    This is used to visualize curves (including BSPLINE-encoded fillets) without
    collapsing them into a single straight segment.
    """
    num_points = max(2, int(num_points))

    if not hasattr(edge, "wrapped"):
        return None

    try:
        from OCP.BRepAdaptor import BRepAdaptor_Curve

        adaptor = BRepAdaptor_Curve(edge.wrapped)
        first = float(adaptor.FirstParameter())
        last = float(adaptor.LastParameter())

        if abs(last - first) < 1e-12:
            p = adaptor.Value(first)
            yz = (float(p.Y()), float(p.Z()))
            return [yz, yz]

        pts: List[Tuple[float, float]] = []
        for i in range(num_points):
            u = first + (last - first) * (i / (num_points - 1))
            p = adaptor.Value(u)
            pts.append((float(p.Y()), float(p.Z())))
        return pts
    except Exception:
        return None


def extract_edge_geometry(edge: Edge) -> EdgeInfo:
    """Extract geometry information from an edge."""
    start_pt = edge.start_point()
    end_pt = edge.end_point()
    
    start_yz = (start_pt.Y, start_pt.Z)
    end_yz = (end_pt.Y, end_pt.Z)
    
    geom_type = edge.geom_type
    
    if geom_type == GeomType.CIRCLE:
        # Get arc center and radius
        center = edge.arc_center
        radius = edge.radius
        
        center_yz = (center.Y, center.Z)
        
        # Calculate start and end angles
        import math
        start_angle = math.degrees(math.atan2(
            start_pt.Z - center.Z, 
            start_pt.Y - center.Y
        ))
        end_angle = math.degrees(math.atan2(
            end_pt.Z - center.Z, 
            end_pt.Y - center.Y
        ))
        
        return EdgeInfo(
            edge_type='ARC',
            start=start_yz,
            end=end_yz,
            center=center_yz,
            radius=radius,
            start_angle=start_angle,
            end_angle=end_angle,
            samples=_sample_edge_points_yz(edge),
        )

    if geom_type == GeomType.LINE:
        return EdgeInfo(
            edge_type='LINE',
            start=start_yz,
            end=end_yz,
            samples=[start_yz, end_yz],
        )

    # Fallback: keep curve as-is and provide sampled points if possible
    return EdgeInfo(
        edge_type='CURVE',
        start=start_yz,
        end=end_yz,
        samples=_sample_edge_points_yz(edge),
    )


def _polyline_length(pts: List[Tuple[float, float]]) -> float:
    total = 0.0
    for a, b in zip(pts, pts[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def _max_distance_to_line(pts: List[Tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0

    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    dx = x1 - x0
    dy = y1 - y0
    denom = math.hypot(dx, dy)
    if denom < 1e-12:
        return 0.0

    max_d = 0.0
    for x, y in pts[1:-1]:
        # Distance to infinite line through (x0,y0)-(x1,y1)
        d = abs((x - x0) * dy - (y - y0) * dx) / denom
        if d > max_d:
            max_d = d
    return max_d


def _solve_3x3(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve 3x3 linear system with partial pivoting."""
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("Singular system")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        inv = 1.0 / m[col][col]
        for j in range(col, 4):
            m[col][j] *= inv

        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, 4):
                m[r][j] -= factor * m[col][j]

    return [m[i][3] for i in range(3)]


def _fit_circle_2d(pts: List[Tuple[float, float]]) -> Optional[Tuple[Tuple[float, float], float, float]]:
    """
    Fit a circle to 2D points using algebraic least squares.

    Returns (center, radius, rel_rms_error) if successful.
    """
    if len(pts) < 5:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # Normal equation components for: x^2 + y^2 + D x + E y + F = 0
    s_x = sum(xs)
    s_y = sum(ys)
    s_xx = sum(x * x for x in xs)
    s_yy = sum(y * y for y in ys)
    s_xy = sum(x * y for x, y in pts)

    s_b = sum(-(x * x + y * y) for x, y in pts)
    s_xb = sum(-x * (x * x + y * y) for x, y in pts)
    s_yb = sum(-y * (x * x + y * y) for x, y in pts)

    a = [
        [s_xx, s_xy, s_x],
        [s_xy, s_yy, s_y],
        [s_x, s_y, float(len(pts))],
    ]
    b = [s_xb, s_yb, s_b]

    try:
        d, e, f = _solve_3x3(a, b)
    except Exception:
        return None

    cx = -d / 2.0
    cy = -e / 2.0
    r2 = cx * cx + cy * cy - f
    if r2 <= 0:
        return None

    r = math.sqrt(r2)
    if not math.isfinite(r) or r < 1e-9:
        return None

    # Relative RMS residual
    residuals = []
    for x, y in pts:
        residuals.append(abs(math.hypot(x - cx, y - cy) - r))
    rms = math.sqrt(sum(rr * rr for rr in residuals) / len(residuals))
    rel_rms = rms / r

    return ( (cx, cy), r, rel_rms )


def _shift_edge_info(info: EdgeInfo, dy: float, dz: float) -> EdgeInfo:
    def shift_pt(pt: Tuple[float, float]) -> Tuple[float, float]:
        return (pt[0] - dy, pt[1] - dz)

    samples = None
    if info.samples:
        samples = [shift_pt(p) for p in info.samples]

    center = None
    if info.center is not None:
        center = shift_pt(info.center)

    return EdgeInfo(
        edge_type=info.edge_type,
        start=shift_pt(info.start),
        end=shift_pt(info.end),
        center=center,
        radius=info.radius,
        start_angle=info.start_angle,
        end_angle=info.end_angle,
        samples=samples,
    )


def _classify_edge_geometry(
    info: EdgeInfo,
    line_tol: float = 1e-4,
    arc_rel_rms_tol: float = 1e-2,
    max_fillet_radius: float = 2.0,
) -> EdgeInfo:
    pts = info.samples or [info.start, info.end]

    # Preserve explicit classification from build123d when available
    if info.edge_type == "LINE":
        return EdgeInfo(edge_type="LINE", start=info.start, end=info.end, samples=pts)
    if info.edge_type == "ARC" and info.center is not None and info.radius is not None:
        return EdgeInfo(
            edge_type="ARC",
            start=info.start,
            end=info.end,
            center=info.center,
            radius=info.radius,
            samples=pts,
        )

    if len(pts) >= 3 and _max_distance_to_line(pts) <= line_tol:
        return EdgeInfo(edge_type="LINE", start=info.start, end=info.end, samples=pts)

    fit = _fit_circle_2d(pts)
    if fit:
        center, radius, rel_rms = fit
        if radius <= max_fillet_radius and rel_rms <= arc_rel_rms_tol:
            return EdgeInfo(
                edge_type="ARC",
                start=info.start,
                end=info.end,
                center=center,
                radius=radius,
                samples=pts,
            )

    return EdgeInfo(edge_type="CURVE", start=info.start, end=info.end, samples=pts)


def _extract_params_from_edge_infos(edge_infos: List[EdgeInfo]) -> ProfileParams:
    # Gather points to determine origin/scale in Y-Z
    all_pts: List[Tuple[float, float]] = []
    for info in edge_infos:
        all_pts.extend(info.samples or [info.start, info.end])

    if not all_pts:
        raise ValueError("No edge sample points to analyze")

    origin_y = min(p[0] for p in all_pts)
    origin_z = min(p[1] for p in all_pts)

    shifted = [_shift_edge_info(info, origin_y, origin_z) for info in edge_infos]
    classified = [_classify_edge_geometry(info) for info in shifted]

    shifted_pts: List[Tuple[float, float]] = []
    for info in classified:
        shifted_pts.extend(info.samples or [info.start, info.end])

    y_max = max(p[0] for p in shifted_pts)
    z_max = max(p[1] for p in shifted_pts)

    # Helpers for line selection
    def is_horizontal_line(e: EdgeInfo, tol: float = 1e-3) -> bool:
        return e.edge_type == "LINE" and abs(e.start[1] - e.end[1]) <= tol

    def is_vertical_line(e: EdgeInfo, tol: float = 1e-3) -> bool:
        return e.edge_type == "LINE" and abs(e.start[0] - e.end[0]) <= tol

    def line_len(e: EdgeInfo) -> float:
        return math.hypot(e.end[0] - e.start[0], e.end[1] - e.start[1])

    z_tol = 1e-3
    y_tol = 1e-3

    # Top outer segment (L2) at z ~= z_max
    top_lines = [e for e in classified if is_horizontal_line(e) and abs((e.start[1] + e.end[1]) / 2 - z_max) <= z_tol]
    if not top_lines:
        raise ValueError("Failed to find top horizontal segment")
    top = max(top_lines, key=lambda e: max(e.start[0], e.end[0]))
    l2 = abs(top.end[0] - top.start[0])
    p1_y = max(top.start[0], top.end[0])
    p0_y = min(top.start[0], top.end[0])

    # Vertical drop from P1 (L3)
    p1 = (p1_y, z_max)
    v_lines = [e for e in classified if is_vertical_line(e) and abs(e.start[0] - p1_y) <= y_tol and abs(max(e.start[1], e.end[1]) - z_max) <= z_tol]
    if not v_lines:
        raise ValueError("Failed to find vertical segment from top-right")
    v1 = max(v_lines, key=line_len)
    l3 = abs(v1.end[1] - v1.start[1])

    # Outer right vertical from bottom (L6) at y ~= y_max, starting at z ~= 0
    right_verticals = [
        e for e in classified
        if is_vertical_line(e)
        and abs(e.start[0] - y_max) <= y_tol
        and abs(min(e.start[1], e.end[1])) <= z_tol
    ]
    if not right_verticals:
        raise ValueError("Failed to find outer-right vertical segment")
    rv = max(right_verticals, key=line_len)
    l6 = abs(rv.end[1] - rv.start[1])
    z_l6 = max(rv.start[1], rv.end[1])

    # Bottom horizontal from outer-right (L5) at z ~= 0 with endpoint at y ~= y_max
    bottom_lines = [
        e for e in classified
        if is_horizontal_line(e)
        and abs(min(e.start[1], e.end[1])) <= z_tol
        and abs(max(e.start[0], e.end[0]) - y_max) <= y_tol
    ]
    if not bottom_lines:
        raise ValueError("Failed to find bottom horizontal segment")
    bottom = max(bottom_lines, key=line_len)
    l5 = abs(bottom.end[0] - bottom.start[0])
    inner_bottom_y = min(bottom.start[0], bottom.end[0])

    # Vertical rise from inner bottom-right (L7)
    inner_verticals = [
        e for e in classified
        if is_vertical_line(e)
        and abs(e.start[0] - inner_bottom_y) <= y_tol
        and abs(min(e.start[1], e.end[1])) <= z_tol
    ]
    if not inner_verticals:
        raise ValueError("Failed to find inner bottom-right vertical segment")
    iv = max(inner_verticals, key=line_len)
    l7 = abs(iv.end[1] - iv.start[1])

    # Upper step horizontal near z ~= L6 with endpoint at y ~= y_max (L4)
    upper_lines = [
        e for e in classified
        if is_horizontal_line(e)
        and abs((e.start[1] + e.end[1]) / 2 - z_l6) <= z_tol
        and abs(max(e.start[0], e.end[0]) - y_max) <= y_tol
    ]
    if not upper_lines:
        raise ValueError("Failed to find upper step horizontal segment")
    upper = max(upper_lines, key=line_len)
    l4 = abs(upper.end[0] - upper.start[0])

    # Detect fillet arcs (R1, R2) by circle-fit + location
    fillet_arcs = []
    for e in classified:
        if e.edge_type != "ARC" or not e.samples or not e.radius:
            continue
        if e.radius <= 0 or e.radius > 1.0:
            continue
        arc_len = _polyline_length(e.samples)
        if e.radius > 1e-9:
            ratio = arc_len / e.radius
            if abs(ratio - (math.pi / 2)) > 0.4:
                continue
        fillet_arcs.append(e)

    r1 = 0.0
    r2 = 0.0
    shelf_inner_y = None
    shelf_z = None
    corner_z = None

    if fillet_arcs:
        # R2 touches the left wall (y ~= 0 after shifting)
        r2_cands = [a for a in fillet_arcs if min(p[0] for p in a.samples) <= 5e-3]
        r1_cands = [a for a in fillet_arcs if a not in r2_cands]

        if r2_cands:
            r2_arc = max(r2_cands, key=lambda a: a.radius or 0.0)
            r2 = float(r2_arc.radius or 0.0)
            corner_z = max(p[1] for p in r2_arc.samples)

        if r1_cands:
            r1_arc = max(r1_cands, key=lambda a: a.radius or 0.0)
            r1 = float(r1_arc.radius or 0.0)
            shelf_inner_y = max(p[0] for p in r1_arc.samples)
            corner_z = min(p[1] for p in r1_arc.samples) if corner_z is None else corner_z

    # Estimate inner shelf anchor (shelf_inner_y, shelf_z) if possible
    if shelf_inner_y is not None:
        shelf_z = max(p[1] for p in shifted_pts if abs(p[0] - shelf_inner_y) <= 5e-3)

    # L8: horizontal segment at shelf_z starting at shelf_inner_y
    l8 = 0.0
    if shelf_inner_y is not None and shelf_z is not None:
        shelf_lines = [
            e for e in classified
            if is_horizontal_line(e)
            and abs((e.start[1] + e.end[1]) / 2 - shelf_z) <= 5e-3
            and (abs(e.start[0] - shelf_inner_y) <= 5e-3 or abs(e.end[0] - shelf_inner_y) <= 5e-3)
        ]
        if shelf_lines:
            shelf_line = max(shelf_lines, key=line_len)
            l8 = abs(shelf_line.end[0] - shelf_line.start[0])

    # L9: vertical distance from shelf_z to the inner corner level (corner_z)
    l9 = 0.0
    if shelf_z is not None and corner_z is not None:
        l9 = max(0.0, shelf_z - corner_z)

    # L10: left vertical segment above the bottom corner (fillet or sharp)
    l10 = 0.0
    if corner_z is not None:
        target_z0 = corner_z - r2 if r2 > 0 else corner_z
        left_verticals = [
            e for e in classified
            if is_vertical_line(e)
            and abs(e.start[0]) <= 5e-3
            and abs(min(e.start[1], e.end[1]) - target_z0) <= 5e-3
        ]
        if left_verticals:
            lv = max(left_verticals, key=line_len)
            l10 = abs(lv.end[1] - lv.start[1])

    # If some values couldn't be derived, fall back to safe defaults from heuristics
    if shelf_inner_y is None:
        shelf_inner_y = 0.6
    if shelf_z is None:
        shelf_z = 1.744
    if l8 <= 0:
        l8 = 0.1
    if l9 <= 0:
        l9 = 0.1
    if l10 <= 0:
        # Prefer existing bottom_z if derivation failed
        l10 = 0.3

    profile_params = ProfileParams(
        L1=round(p0_y, 3),
        L2=round(l2, 3),
        L3=round(l3, 3),
        L4=round(l4, 3),
        L5=round(l5, 3),
        L6=round(l6, 3),
        L7=round(l7, 3),
        L8=round(l8, 3),
        L9=round(l9, 3),
        L10=round(l10, 3),
        R1=round(r1, 3),
        R2=round(r2, 3),
        min_thickness=0.2,
        outer_right_y=round(y_max, 4),
        outer_top_z=round(z_max, 4),
        shelf_inner_y=round(shelf_inner_y, 4),
        shelf_z=round(shelf_z, 4),
        origin_y=round(origin_y, 6),
        origin_z=round(origin_z, 6),
    )

    return profile_params


def get_vertex_coords(v: Vertex) -> Tuple[float, float, float]:
    """Get vertex coordinates rounded to precision."""
    p = v.center()
    return (round(p.X, 4), round(p.Y, 4), round(p.Z, 4))


def analyze_step_file(file_path: Path, output_image_path: Optional[Path] = None) -> ProfileParams:
    """
    Analyze STEP file and extract profile parameters.
    
    Args:
        file_path: Input STEP file path
        output_image_path: Optional path to save dimensioned image
    """
    logger.info(f"Analyzing STEP file: {file_path}")
    
    try:
        shape = import_step(str(file_path))
    except Exception as e:
        logger.error(f"Failed to import STEP: {e}")
        raise ValueError(f"Could not import STEP file: {file_path}")

    # Extract wire (prefer the longest one if multiple exist)
    if isinstance(shape, Wire):
        wires = [shape]
    elif hasattr(shape, "wires"):
        wires = list(shape.wires())
    else:
        raise ValueError("Imported STEP object has no wires()")

    if not wires:
        raise ValueError("No wires found in STEP file")

    def wire_length(w: Wire) -> float:
        try:
            return float(getattr(w, "length"))
        except Exception:
            pass
        try:
            return float(sum(getattr(e, "length", 0.0) for e in w.edges()))
        except Exception:
            return float(len(w.edges()))

    def wire_planarity_x_range(w: Wire) -> float:
        try:
            edges = list(getattr(w, "order_edges")())
        except Exception:
            edges = list(w.edges())

        xs: List[float] = []
        for e in edges:
            try:
                sp = e.start_point()
                ep = e.end_point()
                xs.append(float(sp.X))
                xs.append(float(ep.X))
            except Exception:
                continue

        if not xs:
            return float("inf")
        return max(xs) - min(xs)

    # Prefer wires that are planar in Y-Z (X ~= constant), then the longest perimeter.
    wire = min(wires, key=lambda w: (wire_planarity_x_range(w), -wire_length(w)))

    ordered_edges = wire.order_edges()
    if not ordered_edges:
        raise ValueError("No ordered edges found")

    # Extract edge geometries for visualization
    edge_infos = [extract_edge_geometry(e) for e in ordered_edges]

    vertices = []
    for edge in ordered_edges:
        vertices.append(edge.start_point())
    
    # 2. Find Anchor P1 (Top-Right)
    max_z = -float('inf')
    for v in vertices:
        if v.Z > max_z:
            max_z = v.Z
            
    top_candidates = [i for i, v in enumerate(vertices) if abs(v.Z - max_z) < 0.01]
    
    if not top_candidates:
        p1_idx = 0
    else:
        p1_idx = top_candidates[0]
        max_y = vertices[p1_idx].Y
        for idx in top_candidates:
            if vertices[idx].Y > max_y:
                max_y = vertices[idx].Y
                p1_idx = idx

    vp = vertices[p1_idx:] + vertices[:p1_idx]
    
    # 3. Check Direction
    p1 = vp[0]
    p_next = vp[1] if len(vp) > 1 else vp[0]
    vec = p_next - p1
    
    if abs(vec.Z) < abs(vec.Y):
        logger.info("Wire direction seems reversed (P1->TopLeft), flipping...")
        vp = [vp[0]] + list(reversed(vp[1:]))
    
    num_v = len(vp)
    logger.info(f"Identified {num_v} vertices extracted from wire")

    profile_params = _extract_params_from_edge_infos(edge_infos)

    # Visualization
    if output_image_path and num_v >= 3:
        try:
            from .visualizer import plot_profile_dims
            plot_profile_dims(vp, profile_params, output_image_path, edge_infos)
            logger.info(f"Saved dimension image to: {output_image_path}")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            # Do not fail extraction just because viz failed

    return profile_params
