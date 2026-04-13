import os
import nox


IN_CONTAINER = os.path.exists("/.dockerenv") or os.environ.get("CONTAINER", "") != ""


@nox.session(python=False if IN_CONTAINER else "3.11")
def tests(session):
    """Run the full test suite.

    - Docker/devcontainer: deps are pre-installed via requirements.txt, runs pytest directly.
    - Local: creates an isolated virtualenv and installs from requirements.txt.
    """
    if not IN_CONTAINER:
        session.install("-r", "requirements.txt")
    session.run("pytest", "tests/", "-v", "--tb=short")
