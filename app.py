from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams

model = "microsoft/Phi-3-mini-4k-instruct"
trust_remote_code = True
max_model_len = 512
gpu_memory_utilization = 0.95
enforce_eager = True

llm = LLM(
    model=model,
    trust_remote_code=trust_remote_code,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    enforce_eager=enforce_eager,
)

class RequestItem(BaseModel):
    prompt: str
    output_len: int

class ResponseItem(BaseModel):
    prompt: str
    result: str

app = FastAPI()

@app.post("/run_batch", response_model=List[ResponseItem])
async def run_vllm_endpoint(requests: List[RequestItem]):
    try:
        prompts = [req.prompt for req in requests]
        sampling_params = [
            SamplingParams(
                n=1,
                temperature=0.5,
                top_p=1.0,
                use_beam_search=False,
                ignore_eos=True,
                max_tokens=req.output_len,
            )
            for req in requests
        ]

        results = llm.generate(prompts, sampling_params, use_tqdm=False)
        response = [
            ResponseItem(prompt=req.prompt, result=result.outputs[0].text)
            for req, result in zip(requests, results)
        ]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
