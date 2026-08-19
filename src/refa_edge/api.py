from pathlib import Path
from typing import Literal

import torch

from refa_edge.data import LABEL_NAMES, OPPOSE, QUERY, SUPPORT
from refa_edge.hardware import resolve_device
from refa_edge.models.registry import build_model


def create_app(checkpoint_path: str | Path, device_name: str = "auto"):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "The REST extras are missing. Install them with: pip install -e '.[api]'"
        ) from exc

    class Evidence(BaseModel):
        subject: int = Field(ge=0)
        relation: int = Field(ge=0)
        object: int = Field(ge=0)
        stance: Literal["support", "oppose"]

    class Query(BaseModel):
        subject: int = Field(ge=0)
        relation: int = Field(ge=0)
        object: int = Field(ge=0)

    class PredictRequest(BaseModel):
        evidence: list[Evidence]
        query: Query

    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_name = checkpoint["model_name"]
    model = build_model(
        model_name,
        checkpoint["task_config"],
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    task = checkpoint["task_config"]

    app = FastAPI(
        title="REFA Edge local API",
        version="0.1.0",
        description="Local inference for a trained REFA Edge checkpoint.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model_name, "device": str(device)}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict:
        max_entity = int(task["num_entities"])
        max_relation = int(task["num_relations"])
        all_events = [*request.evidence, request.query]
        for event in all_events:
            if event.subject >= max_entity or event.object >= max_entity:
                raise HTTPException(
                    422, f"Entity IDs must be between 0 and {max_entity - 1}"
                )
            if event.relation >= max_relation:
                raise HTTPException(
                    422, f"Relation IDs must be between 0 and {max_relation - 1}"
                )
        rows = [
            [
                event.subject,
                event.relation,
                event.object,
                SUPPORT if event.stance == "support" else OPPOSE,
                0,
            ]
            for event in request.evidence
        ]
        rows.append(
            [
                request.query.subject,
                request.query.relation,
                request.query.object,
                QUERY,
                1,
            ]
        )
        events = torch.tensor(rows, dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(events), dim=-1)[0].cpu().tolist()
        predicted = int(max(range(3), key=probabilities.__getitem__))
        return {
            "prediction": LABEL_NAMES[predicted],
            "probabilities": {
                LABEL_NAMES[index]: probability
                for index, probability in enumerate(probabilities)
            },
        }

    return app


def serve_checkpoint(
    checkpoint_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "auto",
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "The REST extras are missing. Install them with: pip install -e '.[api]'"
        ) from exc
    uvicorn.run(create_app(checkpoint_path, device), host=host, port=port)
