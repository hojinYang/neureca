from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="neureca",
    version="0.0.1",
    description="A framework for building conversational recommender systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hojin Yang",
    author_email="hojin.yang7@gmail.com",
    url="https://github.com/hojinYang/neureca",
    entry_points={
        "console_scripts": [
            "neureca-train = neureca.cmd:neureca_train_command",
        ],
    },
    install_requires=[
        "click==7.1.2",
        "Flask==1.1.2",
        "joblib==1.0.1",
        "numpy==1.20.2",
        "pandas==1.2.3",
        "pytorch-crf==0.7.2",
        "pytorch-lightning==1.2.7",
        "scikit-learn==0.24.1",
        "scipy==1.6.2",
        "sklearn==0.0",
        "spacy==3.0.6",
        "summarizers==1.0.4",
        "tokenizers==0.10.2",
        "toml==0.10.2",
        "torch==1.8.1",
        "TorchCRF==1.1.0",
        "torchmetrics==0.3.1",
        "tqdm==4.60.0",
        "transformers==4.5.0",
        "typer==0.3.2",
    ],
    packages=find_packages(exclude=["demo-toronto"]),
    python_requires=">=3",
    package_data={"neureca": ["interface/static/*/*", "interface/templates/index.html"]},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
)
