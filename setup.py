from setuptools import setup, find_packages

setup(
    name="autool",
    version="0.1.0",
    description="Efficient tool selection python lib for large language model agents",
    author="Jingyi Jia",
    packages=find_packages(include=["autotool", "autotool.*"]),
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
        "requests>=2.25.0",
        "tqdm>=4.62.0",
        "colorama>=0.4.4",
        "chromadb",
        "sentence_transformers"
    ],
)