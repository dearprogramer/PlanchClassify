# coding=utf-8
from distutils.core import setup
import py2exe
import sys
sys.argv.append('py2exe')

py2exe_options = {

    "includes": ["sip"],  # 如果打包文件中有PyQt代码，则这句为必须添加的

    "dll_excludes": ["MSVCP90.dll", ],  # 这句必须有，不然打包后的程序运行时会报找不到MSVCP90.dll，如果打包过程中找不到这个文件，请安装相应的库

    "compressed": 1,

    "optimize": 2,

    "ascii": 0,

    "bundle_files": 1 # 关于这个参数请看第三部分中的问题(2)
}

setup(
    name='PlanchClassify',
    version='1.0',
    packages=[''],
    url='https://github.com/dearprogramer/cuda_plasma',
    license='TanYao',
    author='Tan Yao && Li Chongwei',
    author_email='510757615@qq.com',
    description='An program to classify the normal and innormal planch',
    #entry_points={'console_scripts': [   'PlanchClassify =main', ]},
    data_files=[('', ['1.jpg'])],
    windows=[{"script": "search.py" } ],  # 括号中更改为你要打包的代码文件名
    zipfile=None,
    options ={"py2exe" : py2exe_options }
)
