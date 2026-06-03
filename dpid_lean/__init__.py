"""DPID + LEAN PBR texture downsample pipeline (no optimization)."""

try:
    from dpid_lean.pipeline import downsample_asset
    __all__ = ["downsample_asset"]
except ModuleNotFoundError:
    __all__ = []
