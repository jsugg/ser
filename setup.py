from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class CustomInstall(install):
    def run(self):
        install.run(self)
        subprocess.check_call(["./ser/configure"])

setup(
    name='ser',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'stable-ts==2.13.3',
        'colored==2.2.3',
        'halo==0.0.31',
        'librosa==0.10.1',
        'numpy==1.26.2',
        'openai-whisper==20231106',
        'ffmpeg-python==0.2.0',
        'scikit-learn==1.3.2',
        'soundfile==0.12.1',
        'typing-extensions==4.8.0'
    ],
    entry_points='''
        [console_scripts]
        ser=ser.__main__:main
    ''',
        cmdclass={
        'runscript': CustomInstall,
    },
    author='Juan Sugg',
    author_email='juanpedrosugg@gmail.com',
    license='MIT',
    keywords='speech emotion recognition',
    url='https://github.com/jsugg/ser/',
    description='A Speech Emotion Recognition tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
