"""FastAPI entry-point skeleton; no model is loaded by this scaffold."""

try:
    from fastapi import FastAPI
except ImportError:  # Allows compile/import checks before optional dependencies are installed.
    FastAPI = None  # type: ignore[assignment,misc]


def create_app() -> object:
    """Create the API shell without loading or training a model."""
    if FastAPI is None:
        raise RuntimeError("Install API dependencies before creating the FastAPI app")
    app = FastAPI(title="Jena Weather Forecasting API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "scaffold"}

    return app


app = create_app() if FastAPI is not None else None
