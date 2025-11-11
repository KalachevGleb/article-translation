from setuptools import setup, find_packages

setup(
    name="article-translator",
    version="0.1.0",
    description="Automated LaTeX article translation using LLM",
    author="Article Translation Team",
    packages=find_packages(),
    install_requires=[
        "openai>=1.12.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "python-dotenv>=1.0.0",
        "jinja2>=3.1.2",
        "pydantic>=2.0.0",
        "tiktoken>=0.5.0",
        "chromadb>=0.4.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "translate-article=article_translator.cli:main",
        ],
    },
)
