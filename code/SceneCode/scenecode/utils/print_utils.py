import contextlib
import os
import sys

from colorama import Fore, Style


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr output, including C-level output.

    This context manager redirects both Python-level I/O (sys.stdout/stderr)
    and OS-level file descriptors (FD 1/2) to /dev/null. This is necessary
    to suppress output from C/C++ libraries like Blender's Cycles renderer,
    which write directly to file descriptors.

    Useful for silencing verbose third-party libraries (e.g., Blender, trimesh).
    Python logging (using logging module) is unaffected.

    Example:
        with suppress_stdout_stderr():
            bpy.ops.render.render(write_still=True)  # No spam!
    """
    # Save original stdout/stderr (Python level).
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Save original file descriptors (OS level).
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    try:
        # Open /dev/null for writing.
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        # Redirect Python stdout/stderr to devnull.
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

        # Redirect OS-level file descriptors to devnull.
        os.dup2(devnull_fd, 1)  # Redirect FD 1 (stdout).
        os.dup2(devnull_fd, 2)  # Redirect FD 2 (stderr).

        # Close devnull_fd as it's no longer needed (duplicated to 1 and 2).
        os.close(devnull_fd)

        yield
    finally:
        # Restore OS-level file descriptors.
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)

        # Close saved file descriptors.
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

        # Restore Python stdout/stderr.
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Fore.RESET}"


def green(x: str) -> str:
    return f"{Fore.GREEN}{x}{Style.RESET_ALL}"


def yellow(x: str) -> str:
    return f"{Fore.YELLOW}{x}{Style.RESET_ALL}"


def bold_green(x: str) -> str:
    return f"{Style.BRIGHT}{Fore.GREEN}{x}{Style.RESET_ALL}"
