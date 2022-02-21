from . import dcgan
from . import sagan
from . import sngan
from . import utils
from .biggan import BigGAN
# from .stylegan import stylegan
from .proggan import proggan
from .biggan_deep import BigGANDeep

# try:
#     from .stylegan2 import stylegan2
# except EnvironmentError:
#     stylegan2 = None
#     print('Warning: could not compile cuda code for stylegan2')
#     print('Ensure $CUDA_HOME/bin/nvcc exists!')
stylegan = None
stylegan2 = None

__all__ = [
    'BigGAN',
    'BigGANDeep',
    'dcgan',
    'sngan',
    'sagan',
    'proggan',
    'stylegan',
    'stylegan2',
    'utils',
]
