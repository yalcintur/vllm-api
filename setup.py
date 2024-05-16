from setuptools import setup, find_packages

setup(
    name="vllm-api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "vllm",
    ],
    entry_points={
        "console_scripts": [
            "vllm-api=main:main",
        ],
    },
)
