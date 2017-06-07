# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='wormbrain',
      version='0.1',
      description='Ising, Boltzmann and entropy analysis tools',
      url='https://github.com/Fuminides/wormbrain',
      author='Javier Fumanal Idocin, Miguel Aguilera',
      author_email='javierfumanalidocin@gmail.com',
      license='MIT',
      packages=['wormbrain'],
      install_requires=[
          'matplotlib',
          'networkx',
          'plotly',
          'sklearn',
          'numba',
          'numpy'  
      ],
      dependency_links = [ 'https://github.com/Fuminides/Python-Entropy-Tools' ],
      zip_safe=False)

