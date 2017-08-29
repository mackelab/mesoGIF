from setuptools import setup

setup(
    name='finite-size-GIF',
    version='0.1dev',
    py_modules=['main',
                'fsgif_model'],
    install_requires=[
        'Click',
        'simpleeval',
        #'theano_shim >= 0.2',
        #'sinn >= 0.1'
    ],
    entry_points='''
        [console_scripts]
        fsgif=main:cli
    ''',
)
