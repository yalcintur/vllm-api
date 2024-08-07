from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from fastapi.responses import JSONResponse, Response, StreamingResponse

model = "HFInternal/altai_tr_v1"
trust_remote_code = True
max_model_len = 2048
gpu_memory_utilization = 0.95
enforce_eager = True

llm = LLM(
    model=model,
    trust_remote_code=trust_remote_code,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    enforce_eager=enforce_eager,
    enable_prefix_caching=True,
    disable_sliding_window=True,
)

class RequestItem(BaseModel):
    prompt: str
    output_len: int

class ResponseItem(BaseModel):
    result: str

app = FastAPI()

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/run_batch", response_model=List[ResponseItem])
async def run_vllm_endpoint(requests: List[RequestItem]):
    try:
        prompts = [req.prompt for req in requests]
        sampling_params = [
            SamplingParams(
                n=1,
                temperature=0.3,
                top_p=0.9,
                use_beam_search=False,
                ignore_eos=False,
                max_tokens=req.output_len,
            )
            for req in requests
        ]

        results = llm.generate(prompts, sampling_params, use_tqdm=False)
        response = [
            ResponseItem(result=result.outputs[0].text)
            for result in results
        ]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
