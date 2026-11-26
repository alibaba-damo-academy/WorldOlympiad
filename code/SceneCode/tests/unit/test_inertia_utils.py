import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from scenecode.utils.inertia_utils import ensure_valid_inertia, fix_sdf_file_inertia


class TestEnsureValidInertia(unittest.TestCase):
    """Test ensure_valid_inertia eigenvalue projection."""

    def test_valid_tensor_returned_unchanged(self):
        """A physically valid tensor should be returned as-is."""
        # Uniform sphere: Ixx = Iyy = Izz, products = 0.
        tensor = np.diag([2.0, 2.0, 2.0])
        result = ensure_valid_inertia(tensor)
        np.testing.assert_array_equal(result, tensor)

    def test_valid_non_diagonal_tensor_unchanged(self):
        """A valid tensor with off-diagonal terms should be unchanged."""
        # Construct a valid tensor via known eigenvalues [1, 2, 2.5].
        # Triangle inequality: 1 + 2 = 3 >= 2.5. Valid.
        eigenvalues = np.array([1.0, 2.0, 2.5])
        # Arbitrary orthogonal rotation.
        v = np.linalg.qr(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10.0]]))[0]
        tensor = v @ np.diag(eigenvalues) @ v.T
        result = ensure_valid_inertia(tensor)
        np.testing.assert_array_equal(result, tensor)

    def test_triangle_inequality_violation_fixed(self):
        """A tensor barely violating the triangle inequality gets fixed."""
        # Eigenvalues [0.1, 0.2, 0.5]: 0.1 + 0.2 = 0.3 < 0.5. Violated.
        tensor = np.diag([0.1, 0.2, 0.5])
        result = ensure_valid_inertia(tensor)

        # Check result is valid.
        eigs = np.sort(np.linalg.eigvalsh(result))
        self.assertGreaterEqual(eigs[0] + eigs[1], eigs[2])
        # All eigenvalues positive.
        self.assertTrue(np.all(eigs > 0))

    def test_negative_eigenvalue_fixed(self):
        """A tensor with negative eigenvalues gets clamped."""
        # Build a tensor with one negative eigenvalue.
        eigenvalues = np.array([-0.01, 1.0, 1.5])
        v = np.eye(3)
        tensor = v @ np.diag(eigenvalues) @ v.T
        result = ensure_valid_inertia(tensor)

        eigs = np.sort(np.linalg.eigvalsh(result))
        # All eigenvalues must be positive.
        self.assertTrue(np.all(eigs > 0))
        # Triangle inequality must hold.
        self.assertGreaterEqual(eigs[0] + eigs[1], eigs[2])

    def test_result_is_symmetric(self):
        """Fixed tensor must be symmetric."""
        tensor = np.diag([0.1, 0.2, 0.5])
        result = ensure_valid_inertia(tensor)
        np.testing.assert_array_almost_equal(result, result.T)


class TestFixSdfFileInertia(unittest.TestCase):
    """Test fix_sdf_file_inertia on sample SDF files."""

    def _make_sdf(self, ixx, iyy, izz, ixy=0.0, ixz=0.0, iyz=0.0):
        """Create a minimal SDF string with given inertia values."""
        return f"""\
<?xml version='1.0' encoding='utf-8'?>
<sdf version="1.7">
  <model name="test">
    <link name="base_link">
      <inertial>
        <mass>1.0</mass>
        <pose>0 0 0 0 0 0</pose>
        <inertia>
          <ixx>{ixx:.6e}</ixx>
          <iyy>{iyy:.6e}</iyy>
          <izz>{izz:.6e}</izz>
          <ixy>{ixy:.6e}</ixy>
          <ixz>{ixz:.6e}</ixz>
          <iyz>{iyz:.6e}</iyz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
"""

    def test_valid_sdf_unchanged(self):
        """An SDF with valid inertia should not be modified."""
        sdf_content = self._make_sdf(ixx=2.0, iyy=2.0, izz=2.0)
        with tempfile.NamedTemporaryFile(suffix=".sdf", mode="w", delete=False) as f:
            f.write(sdf_content)
            sdf_path = Path(f.name)

        try:
            modified = fix_sdf_file_inertia(sdf_path)
            self.assertFalse(modified)
        finally:
            sdf_path.unlink()

    def test_invalid_sdf_gets_fixed(self):
        """An SDF with invalid inertia should be fixed in-place."""
        # Triangle inequality violation: 0.1 + 0.2 < 0.5.
        sdf_content = self._make_sdf(ixx=0.1, iyy=0.2, izz=0.5)
        with tempfile.NamedTemporaryFile(suffix=".sdf", mode="w", delete=False) as f:
            f.write(sdf_content)
            sdf_path = Path(f.name)

        try:
            modified = fix_sdf_file_inertia(sdf_path)
            self.assertTrue(modified)

            # Re-parse and verify the fixed values.
            tree = ET.parse(sdf_path)
            root = tree.getroot()
            inertia = root.find(".//inertia")
            ixx = float(inertia.findtext("ixx"))
            iyy = float(inertia.findtext("iyy"))
            izz = float(inertia.findtext("izz"))

            tensor = np.array(
                [
                    [ixx, 0, 0],
                    [0, iyy, 0],
                    [0, 0, izz],
                ]
            )
            eigs = np.sort(np.linalg.eigvalsh(tensor))
            self.assertGreaterEqual(eigs[0] + eigs[1], eigs[2])
        finally:
            sdf_path.unlink()

    def test_mass_preserved_after_fix(self):
        """Mass and pose should be preserved when inertia is fixed."""
        sdf_content = self._make_sdf(ixx=0.1, iyy=0.2, izz=0.5)
        with tempfile.NamedTemporaryFile(suffix=".sdf", mode="w", delete=False) as f:
            f.write(sdf_content)
            sdf_path = Path(f.name)

        try:
            fix_sdf_file_inertia(sdf_path)

            tree = ET.parse(sdf_path)
            root = tree.getroot()
            mass = float(root.findtext(".//mass"))
            pose = root.findtext(".//inertial/pose")

            self.assertEqual(mass, 1.0)
            self.assertEqual(pose.strip(), "0 0 0 0 0 0")
        finally:
            sdf_path.unlink()


if __name__ == "__main__":
    unittest.main()
