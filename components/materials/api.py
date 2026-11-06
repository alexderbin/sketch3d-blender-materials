"""Public entry of Materials (ARCHITECTURE.md: the mechanism stays with its owner, neighbours call api): the shared model material and the weather layer by the
materials plan (mat_plan resolve). Formats use it from here; the core calls Materials only through the passport's extension points
(hooks: frame_materials, weather). The Blender part is imported on first access."""

_BL = ("apply_plan", "cams_from_scene", "cams_from_json")   # bl_mat


def __getattr__(name):
    if name in _BL:
        from components.materials import bl_mat
        return getattr(bl_mat, name)
    raise AttributeError(f"components.materials.api: no '{name}'")
