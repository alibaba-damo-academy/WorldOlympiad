# SceneCode → PyBullet minimal bridge

This directory is intentionally independent of the previous `autoresearch`
implementation. The official SceneCode tree remains read-only.

The bridge creates a physics-only view of an official SceneCode SDF:

- preserves model/link structure, mass, inertia, friction, and collision meshes;
- removes only glTF visual elements that PyBullet cannot parse;
- removes Drake-only `declare_convex` annotations;
- rewrites collision mesh paths to absolute paths without copying or changing them;
- records source and output SHA-256 hashes.

RGB presentation is not produced by PyBullet. It remains the responsibility of
the official SceneCode/Blender visual path.

The first end-to-end sample is intentionally narrow: a PyBullet-bundled Franka
Panda picks up the official SceneCode blue cup and places it in the official
SceneCode blue sorting tray on the generated worktable. PyBullet is the sole
physics/state authority. The robot is explicitly recorded as an external
execution backend because SceneCode does not publish a robot embodiment or
low-level controller.
