FROM verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

RUN export NVTE_FRAMEWORK=pytorch && pip3 install boto3