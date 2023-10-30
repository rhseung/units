from setuptools import setup, find_packages

setup(
	name='rhseung.units',
	version='0.3',
	description='A package for handling physics units.',
	author='rhseung',
	license='MIT',
	author_email='rhseungg@gmail.com',
	url='https://github.com/rhseung/units',
	packages=find_packages(),
	keywords=['units', 'physics', 'mathematics'],
	python_requires='>=3.10',
	package_data={},
	zip_safe=False,
)
