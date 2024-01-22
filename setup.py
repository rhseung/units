from setuptools import setup, find_packages
from unit import __package_name__, __version__

with open('README.md', 'r', encoding='utf-8') as f:
	long_description = f.read()
with open('requirements.txt', 'r', encoding='utf-8') as f:
	requirements = f.read().splitlines()
with open('requirements-test.txt', 'r', encoding='utf-8') as f:
	test_requirements = f.read().splitlines()

setup(
	name=__package_name__,
	version=__version__,
	long_description=long_description,
	install_requires=requirements,
	test_requirements=test_requirements,
	python_requires='>=3.11',
	packages=find_packages(),
	author='rhseung',
	author_email='rhseungg@gmail.com',
	url='https://github.com/rhseung/units',
	keywords=['units', 'physics', 'mathematics'],
)
