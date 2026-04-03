from pathlib import Path

_root_dir = Path(__file__).resolve().parents[2]
_evoar_pkg_dir = _root_dir / "EvoAR-v1" / "autoregressive"
if _evoar_pkg_dir.is_dir():
    _evoar_pkg = str(_evoar_pkg_dir)
    if _evoar_pkg not in __path__:
        __path__.append(_evoar_pkg)
