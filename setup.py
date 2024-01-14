import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="sustech_ncs",
  version="0.0.2",
  author="Yibo Yang, Jimao Shi",
  author_email="12112222@mail.sustech.edu.cn, 12112218@mail.sustech.edu.cn",
  description="A python impletation of NCS-C.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/Joy-Mine/SUSTech-NCS",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)