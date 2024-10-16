from setuptools import setup, find_packages

setup(
    name="atspeed",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
    ],
    author="AtSpeed team",
    author_email="你的邮箱",
    description="包的描述",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/你的用户名/my_package",  # 项目的URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)