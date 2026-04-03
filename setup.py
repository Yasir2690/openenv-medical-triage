from setuptools import setup, find_packages

setup(
    name="medical-triage-env",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "openenv-server = server.app:start_server",
        ],
    },
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "python-dateutil>=2.8.2",
        "gradio>=4.0.0",
        "pandas>=2.0.0",
        "openai>=1.0.0",
        "uvicorn>=0.30.0",
        "fastapi>=0.115.0",
        "openenv-core>=0.2.0",
    ],
)