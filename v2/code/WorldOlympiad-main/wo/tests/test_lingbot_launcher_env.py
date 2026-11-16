import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from wo.models.lingbot_va import _server_pythonpath


class LingbotLauncherEnvironmentTests(unittest.TestCase):
    def test_site_packages_are_forwarded_for_symlinked_venv_launcher_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv = root / "venv"
            launcher = venv / "bin/python"
            launcher.parent.mkdir(parents=True)
            launcher.symlink_to(sys.executable)
            (venv / "pyvenv.cfg").write_text("home = /shared/python\n")
            site = venv / "lib/python3.11/site-packages"
            site.mkdir(parents=True)
            (site / "venv_probe.py").write_text("VALUE = 'found'\n")
            robotwin = root / "robotwin"
            lingbot = root / "lingbot"
            robotwin.mkdir()
            lingbot.mkdir()

            pythonpath = _server_pythonpath(launcher, robotwin, lingbot)
            self.assertIn(str(site), pythonpath.split(os.pathsep))
            child_env = dict(os.environ, PYTHONPATH=pythonpath)
            result = subprocess.run(
                [str(launcher.resolve()), "-c", "import venv_probe; print(venv_probe.VALUE)"],
                check=True, capture_output=True, text=True, env=child_env,
            )
            self.assertEqual(result.stdout.strip(), "found")

    def test_site_packages_are_not_added_for_non_venv_launchers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / "bin/python"
            launcher.parent.mkdir()
            launcher.write_text("not a real interpreter")
            first, second = root / "robotwin", root / "lingbot"
            expected = os.pathsep.join((str(first), str(second), str(first / "scripts")))
            self.assertEqual(_server_pythonpath(launcher, first, second), expected)


if __name__ == "__main__":
    unittest.main()
