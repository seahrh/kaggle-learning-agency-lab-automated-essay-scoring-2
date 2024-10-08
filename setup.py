from setuptools import find_packages, setup

__version__ = "0.1"
setup(
    name="lalaes2",
    version=__version__,
    python_requires=">=3.9,<3.13",
    install_requires=[
        "lightgbm==4.2.0",
        "pandas==2.2.2",
        "pyarrow==16.1.0",
        "scikit-learn==1.5.0",
        "sentencepiece==0.2.0",
        "spacy==3.7.4",
        "protobuf==3.20.3",  # required for deberta-v3 tokenizer
        "transformers==4.41.2",
        "pytorch-lightning==2.2.5",
        "tqdm==4.66.4",
        "accelerate==0.30.1",
        "bitsandbytes==0.43.1",
        "textstat==0.7.3",
    ],
    extras_require={
        "embeddings": [
            "faiss-cpu==1.7.4",
            "sentence-transformers==2.2.2",
        ],
        "lint": [
            "black==24.2.0",
            "isort==5.13.2",
            "pre-commit==3.6.1",
            "flake8==7.0.0",
            "mypy==1.8.0",
        ],
        "tests": [
            "pytest==7.4.4",
            "pytest-cov==4.1.0",
        ],
        "notebook": ["jupyterlab==4.0.11", "ipywidgets==8.1.1", "seaborn==0.12.2"],
    },
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    description="Learning Agency Lab - Automated Essay Scoring 2.0. 2024 Kaggle Competition",
    license="MIT",
    author="seahrh",
    author_email="seahrh@gmail.com",
    url="https://github.com/seahrh/kaggle-learning-agency-lab-automated-essay-scoring-2",
)
