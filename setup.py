from setuptools import setup


requires = ['numpy','os','shutil','time']

packages = ['fastMLP',]

package_dir = {'fastMLP' : 'fastMLP'}
package_data = { 'fastMLP' : []}


setup(
    name='optinpy',
    version='1.0.0',
    packages=packages,
    license='The  GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 License',
    author = 'Gabriel S. Gusmao',
    author_email = 'gusmaogabriels@gmail.com',
    url = 'https://github.com/gusmaogabriels/fastMLP',
    download_url = 'https://github.com/gusmaogabriels/fastMLP/tarball/v1.0.0',
    keywords = ['python', 'optimization', 'ANN', 'MLPr', 'perceptron', 'neural', 'network'],
    package_data = package_data,
    package_dir = package_dir,

)
