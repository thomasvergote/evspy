from setuptools import setup

# load the README file and use it as the long_description for PyPI
def readme():
    with open('README.md', 'r') as f:
        return f.read()

setup(name='evspy',
      version='0.1.1',
      description='Modelling of consolidation with creep and swelling of soils',
      long_description=readme(),
      url='http://github.com/thomasvergote/evspy',
      author='Thomas Vergote',
      author_email='thomas@inferensics.be',
      license='GPL v3',
      packages=['evspy'],
      keywords=['engineering', 'geotechnical'],
      classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
      ],
      zip_safe=False)
