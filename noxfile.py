import os
import nox


IN_CONTAINER = os.path.exists("/.dockerenv") or os.environ.get("CONTAINER", "") != ""


@nox.session(python=False)
def tests(session):
    """Run the full test suite.

    - Docker/devcontainer: deps are pre-installed via requirements.txt, runs pytest directly.
    - Local: creates an isolated virtualenv and installs from requirements.txt.
    """
    if not IN_CONTAINER:
        session.run("pip", "install", "-r", "requirements.txt")
    session.run("pytest", "tests/", "-v", "--tb=short")
