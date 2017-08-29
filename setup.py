from setuptools import setup

setup(
    name='finite-size-GIF',
    version='0.1dev',
    py_modules=['generate_input',
                'generate_spikes',
                'generate_activity',
                'fsgif_model'],
    install_requires=[
        'Click',
        'simpleeval',
        #'theano_shim >= 0.2',
        #'sinn >= 0.1'
    ],
    dependency_links=[
        'git+ssh://git@github.com:alcrene/parameters.git',
    ],
    entry_points='''
        [console_scripts]
        fsgif=main:cli
    ''',
)
