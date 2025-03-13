import nox

nox.options.default_venv_backend = "uv"


@nox.session
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", "src")
    session.run("ruff", "check", "tests")


def fmt(session):
    session.install("ruff")
    session.run("ruff", "format", "--check")


@nox.session(python="3.10")
def tests(session):
    session.env.update({"UV_PROJECT_ENVIRONMENT": session.virtualenv.location})
    session.run("uv", "sync", "--all-extras", "--dev")
    session.run("bash", "./scripts/install_cdo.sh", external=True)
    session.run("uv", "run", "pytest", "--cov", "--cov-report=html")
