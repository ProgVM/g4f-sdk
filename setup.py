import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

REQUIRED_PACKAGES = [
  'g4f',
  'tiktoken'
]

setuptools.setup(
  name="g4f-sdk",
  version="0.1.1",
  author="ProgVM",
  author_email="progvminc@example.com",
  description="The missing resilient and intelligent SDK for the g4f (GPT4Free) library.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/ProgVM/g4f-sdk",
  packages=setuptools.find_packages(),
  install_requires=REQUIRED_PACKAGES,
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
  ],
  python_requires='>=3.9',
)