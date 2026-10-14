"""Area-of-interest parsing: any location as a bbox, a point + radius, or a Bhoonidhi tile ID.

Accepted strings (whitespace-insensitive):

* ``bbox:91.6,26.0,91.9,26.3`` or ``91.6,26.0,91.9,26.3`` -- min_lon, min_lat, max_lon, max_lat (EPSG:4326)
* ``bbox:...@EPSG:32646`` -- bbox in another CRS
* ``point:26.14,91.74,5km`` or ``26.14,91.74,5km`` -- lat, lon, radius (``km`` or ``m``; default km)
* ``tile:R2A_L4_104_053`` -- Bhoonidhi tile/scene ID, resolved to a footprint by a ``DataSource``
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pyproj import CRS, Transformer
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

AOIKind = Literal["bbox", "point", "tile"]
WGS84 = "EPSG:4326"


class AOIError(ValueError):
    pass


@dataclass(frozen=True)
class AOI:
    kind: AOIKind
    geometry: BaseGeometry | None
    crs: str
    tile_id: str | None = None
    text: str = ""

    def to_crs(self, crs: str | CRS) -> BaseGeometry:
        if self.geometry is None:
            raise AOIError(f"AOI {self.text!r} has no geometry (unresolved tile ID)")
        dst = CRS.from_user_input(crs)
        src = CRS.from_user_input(self.crs)
        if src == dst:
            return self.geometry
        tr = Transformer.from_crs(src, dst, always_xy=True)
        return shp_transform(tr.transform, self.geometry)


_NUM = r"[-+]?\d+(?:\.\d+)?"


def _floats(s: str) -> list[float]:
    return [float(x) for x in re.findall(_NUM, s)]


def _point_buffer(lat: float, lon: float, radius_m: float) -> BaseGeometry:
    """Exact-distance circle: buffer in a local azimuthal-equidistant projection, back to WGS84."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise AOIError(f"lat/lon out of range: {lat}, {lon}")
    if radius_m <= 0:
        raise AOIError("radius must be positive")
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    back = Transformer.from_crs(aeqd, WGS84, always_xy=True)
    circle = Point(0.0, 0.0).buffer(radius_m, quad_segs=32)
    return shp_transform(back.transform, circle)


def parse_aoi(
    text: str,
    tile_resolver: Callable[[str], tuple[BaseGeometry, str] | None] | None = None,
) -> AOI:
    """Parse an AOI string. ``tile_resolver(tile_id) -> (geometry, crs)`` resolves tile IDs."""
    raw = text.strip()
    if not raw:
        raise AOIError("empty AOI")
    low = raw.lower()

    bare = not low.startswith(("tile:", "bbox:", "point:"))
    if low.startswith("tile:") or (bare and "," not in raw):
        tile_id = raw.split(":", 1)[1].strip() if low.startswith("tile:") else raw
        if not re.fullmatch(r"[A-Za-z0-9_\-.]+", tile_id):
            raise AOIError(f"invalid tile ID: {tile_id!r}")
        geom, crs = (None, WGS84)
        if tile_resolver is not None:
            resolved = tile_resolver(tile_id)
            if resolved is not None:
                geom, crs = resolved
        return AOI("tile", geom, crs, tile_id=tile_id, text=raw)

    body = raw.split(":", 1)[1] if low.startswith(("bbox:", "point:")) else raw
    crs = WGS84
    if "@" in body:
        body, crs_txt = body.split("@", 1)
        crs = CRS.from_user_input(crs_txt.strip()).to_string()
    nums = _floats(body)

    if low.startswith("bbox:") or (not low.startswith("point:") and len(nums) == 4):
        if len(nums) != 4:
            raise AOIError("bbox needs 4 numbers: min_x,min_y,max_x,max_y")
        x0, y0, x1, y1 = nums
        if x0 >= x1 or y0 >= y1:
            raise AOIError("bbox min must be < max")
        if crs == WGS84 and not (
            -180 <= x0 <= 180 and -180 <= x1 <= 180 and -90 <= y0 <= 90 and -90 <= y1 <= 90
        ):
            raise AOIError("bbox out of lon/lat range")
        return AOI("bbox", box(x0, y0, x1, y1), crs, text=raw)

    if len(nums) == 3:
        lat, lon, r = nums
        unit_m = bool(re.search(r"\d\s*m\s*$", body.strip().lower())) and not body.strip().lower().endswith(
            "km"
        )
        radius_m = r if unit_m else r * 1000.0
        return AOI("point", _point_buffer(lat, lon, radius_m), WGS84, text=raw)

    raise AOIError(f"cannot parse AOI {text!r}")
