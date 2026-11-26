from collections import Counter

from wo.tasks.closed_box_export import closed_box_mesh, visual_to_collision_axes


def _closed_box_faces(ring_count=32):
    # Compatibility alias for historical topology assertions; production and
    # regression code share the actual source-builder boundary now.
    return closed_box_mesh(ring_count)


def test_closed_box_source_uses_closed_shell_revision():
    counts = Counter(
        tuple(sorted(edge))
        for face in _closed_box_faces()
        for edge in zip(face, face[1:] + face[:1])
    )
    assert counts and set(counts.values()) == {2}
    assert max(v for edge in counts for v in edge) <= 65


def test_export_axes_are_not_silently_swapped():
    # A 0.20 x 0.15 x 0.08 Z-up box is represented as X/Y/Z in collision,
    # but its Y-up GLB source has X/Z/Y extents.  This is the production
    # failure mode that previously let visual and collision packages diverge.
    source = [(0.0, 0.0, 0.0), (0.20, 0.08, 0.15)]
    converted = visual_to_collision_axes(source)
    dims = tuple(converted[1][i] - converted[0][i] for i in range(3))
    assert tuple(abs(v) for v in dims) == (0.20, 0.15, 0.08)


def test_export_axes_preserve_signed_translation_and_origin():
    source = [(1.2, -0.4, 2.5), (1.4, -0.1, 2.7)]
    assert visual_to_collision_axes(source) == [(1.2, -2.5, -0.4),
                                                (1.4, -2.7, -0.1)]
