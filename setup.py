from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = ROOT / "README.md"

setup(
    name="memoryos",
    version="0.1.0",
    description="Persistent long-term memory layer for AI applications.",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="Aryan Gupta",
    packages=find_packages(exclude=("tests", "tests.*", "examples", "examples.*")),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
    ],
    extras_require={
        "embeddings": ["sentence-transformers>=2.2.0"],
        "faiss": ["faiss-cpu>=1.7.4"],
        "dev": ["pytest>=7.0", "sentence-transformers>=2.2.0"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
