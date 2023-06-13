from setuptools import setup, find_packages

setup(
    name='ddg_predictor',
    version='1.0.0',
    url='https://github.com/HeliXonProtein/binding-ddg-predictor',
    author='Author Name',
    author_email='luost@helixon.com',
    description="""ddg-predictor predicts changes in binding energy upon mutation (ddG) for protein-protein complexes.""",
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'ddg_predict=ddg_predictor.scripts.predict:main',
        ],
    },
)
