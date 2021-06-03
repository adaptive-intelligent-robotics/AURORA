#! /usr/bin/env python
import os.path
import sys

# Adding Module paths (to take additional waf_tools from subdirs into account)
MODULES_PATH = os.path.abspath(os.path.join(sys.path[0], os.pardir, os.pardir, 'modules'))
for specific_module_folder in os.listdir(MODULES_PATH):
    sys.path.append(os.path.join(MODULES_PATH, specific_module_folder))
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], os.pardir)))

from waflib.Configure import conf
from waflib import Logs

import sferes
import aurora.waf_tools.magnum as magnum
import aurora.waf_tools.magnum_integration as magnum_integration

import aurora.singularity.collections_experiments.maze as maze
import aurora.singularity.collections_experiments.hexapod_camera_vertical as hexapod_camera_vertical
import aurora.singularity.collections_experiments.air_hockey as air_hockey
import aurora.singularity.factory_experiments as factory_experiments

PROJECT_NAME = "aurora"


def get_relative_path(waf_tool_name):
    return PROJECT_NAME + '.' + 'waf_tools' + '.' + waf_tool_name


def options(opt):
    opt.load(get_relative_path('dart'))
    opt.load(get_relative_path('corrade'))
    opt.load(get_relative_path('magnum'))
    opt.load(get_relative_path('magnum_integration'))
    opt.load(get_relative_path('magnum_plugins'))
    opt.load(get_relative_path('robot_dart'))
    opt.load(get_relative_path('robox2d'))

    opt.add_option('--chosen',
                   type='string',
                   help='Name of the group of experiments to build',
                   dest='chosen')

    opt.add_option('--dbg',
                   action='store_true',
                   default=False,
                   help='Activate all DBG logging info',
                   dest='dbg')


@conf
def configure(conf):
    print('conf exp:')
    conf.load(get_relative_path('dart'))
    conf.load(get_relative_path('corrade'))
    conf.load(get_relative_path('magnum'))
    conf.load(get_relative_path('magnum_integration'))
    conf.load(get_relative_path('magnum_plugins'))
    conf.load(get_relative_path('robot_dart'))
    conf.load(get_relative_path('robox2d'))

    conf.check_dart()
    conf.check_corrade(components='Utility PluginManager', required=False)
    conf.env['magnum_dep_libs'] = 'MeshTools Primitives Shaders SceneGraph GlfwApplication'
    if conf.env['DEST_OS'] == 'darwin':
        conf.env['magnum_dep_libs'] += ' WindowlessCglApplication'
    else:
        conf.env['magnum_dep_libs'] += ' WindowlessGlxApplication'
    conf.check_magnum(components=conf.env['magnum_dep_libs'], required=False)
    conf.check_magnum_plugins(components='AssimpImporter', required=False)
    conf.check_magnum_integration(components='Dart', required=False)

    print(conf.env.INCLUDES_MagnumIntegration)
    if len(conf.env.INCLUDES_MagnumIntegration) > 0:
        conf.get_env()['BUILD_MAGNUM'] = True
        conf.env['magnum_libs'] = magnum.get_magnum_dependency_libs(conf, conf.env['magnum_dep_libs']) \
                                  + magnum_integration.get_magnum_integration_dependency_libs(conf, 'Dart')
    print(conf.env['magnum_libs'])
    #  conf.env['magnum_libs'] = conf.env['magnum_libs'] + ' MAGNUM_GL'

    conf.check_robot_dart()
    conf.check_robox2d()

    conf.env.append_unique('LINKFLAGS', '-Wl,--no-as-needed')


def build(bld):
    bld.env.LIBPATH_PYTORCH = '/workspace/lib/torch/'
    # bld.env.LIB_PYTORCH = 'torch c10 c10_cuda caffe2_detectron_ops_gpu caffe2_module_test_dynamic caffe2_nvrtc'.split(
    #    ' ')
    bld.env.LIB_PYTORCH = 'torch_cpu torch_cuda torch_global_deps shm torch c10 c10_cuda'.split(' ')
    bld.env.INCLUDES_PYTORCH = ['/workspace/include/torch', '/workspace/include/torch/torch/csrc/api/include']

    bld.env.LIBPATH_CUDA = '/usr/local/cuda/lib64/'
    # bld.env.LIB_PYTORCH = 'torch c10 c10_cuda caffe2_detectron_ops_gpu caffe2_module_test_dynamic caffe2_nvrtc'.split(
    #    ' ')
    bld.env.LIB_CUDA = 'cudart'.split(' ')
    bld.env.INCLUDES_CUDA = ['/usr/local/cuda/include']

    bld.env.LIBPATH_LIBFASTSIM = ['/workspace/lib']
    bld.env.LIB_LIBFASTSIM = ['fastsim']
    bld.env.INCLUDES_LIBFASTSIM = ['/workspace/include']

    bld.env.LIBPATH_MAGNUMTEXT = ['/workspace/lib']
    bld.env.LIB_MAGNUMTEXT = ['MagnumText']
    bld.env.INCLUDES_MAGNUMTEXT = ['/workspace/include']


    list_experiments_all = factory_experiments.get_all_experiments_from_dict(factory_experiments.DICT_ALL_EXPERIMENTS)

    list_experiments_tests = [
        # air_hockey.AIR_HOCKEY_HAND_CODED_GT,

        air_hockey.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM[10],
        # hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_GT,
    ]

    if bld.options.chosen in factory_experiments.DICT_ALL_EXPERIMENTS:
        Logs.info(f'Building {bld.options.chosen} experiments')
        chosen_experiments = factory_experiments.DICT_ALL_EXPERIMENTS[bld.options.chosen]
    elif bld.options.chosen in ['test', 'tests']:
        Logs.info(f'Building Test experiments')
        chosen_experiments = list_experiments_tests
    elif bld.options.chosen == 'none':
        Logs.info(f'NOT BUILDING ANY EXPERIMENT')
        chosen_experiments = []
    elif bld.options.chosen == 'all':
        Logs.info(f'Building All experiments')
        chosen_experiments = list_experiments_all
    else:
        Logs.warn(f'No chosen experiments to build! -> Building all experiments')
        chosen_experiments = list_experiments_all

    if bld.options.dbg:
        Logs.info('Using DBG library')
        dbg_flag = ' DBG_ENABLED'
    else:
        Logs.info('Not using DBG')
        dbg_flag = ''

    # sferes
    libs = 'BOOST BOOST_FILESYSTEM BOOST_SYSTEM BOOST_SERIALIZATION BOOST_PROGRAM_OPTIONS TBB PTHREAD'
    mpi = bld.env['MPI_ENABLED']
    if mpi:
        libs += ' MPI BOOST_MPI'
    bld.stlib(features = 'cxx cxxstlib',
              source = 'cpp/dbg_tools/dbg.cpp ../../sferes/dbg/dbg.cpp',
              includes = '. dbg ../../',
              uselib = libs,
              target = 'dbg_aurora')

    # building main experiment executables
    sferes.create_variants(bld,
                           source='cpp/aurora.cpp',
                           includes='./cpp . ../../',
                           uselib=bld.env['magnum_libs'] 
                                  + 'ROBOTDART ROBOT_DART TBB BOOST EIGEN PTHREAD MPI'
                                  + ' DART DART_GRAPHIC'
                                  + ' PYTORCH LIBFASTSIM SDL ROBOX2D CUDA MAGNUMTEXT',
                           use='dbg_aurora',
                           target='aurora',
                           # Taking the set of chosen experiments to avoid any duplicate
                           variants=[x.get_str_variables_run_experiment() + dbg_flag for x in set(chosen_experiments)],
                           )

    # Example script for saving video
    sferes.create_variants(bld,
                           source='cpp/save_video.cpp',
                           includes='./cpp . ../../',
                           uselib=bld.env['magnum_libs']
                                  + 'ROBOTDART ROBOT_DART TBB BOOST EIGEN PTHREAD MPI'
                                  + ' DART DART_GRAPHIC'
                                  + ' PYTORCH LIBFASTSIM SDL ROBOX2D MAGNUMTEXT CUDA',
                           use='dbg_aurora',
                           target='save_video',
                           # Taking the set of chosen experiments to avoid any duplicate
                           variants=[hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM[10].get_str_variables_run_experiment(), air_hockey.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM[2].get_str_variables_run_experiment(), air_hockey.AIR_HOCKEY_HAND_CODED_GT.get_str_variables_run_experiment()],
                           )

    # Example script for training AE from scratch as in paper
    sferes.create_variants(bld,
                           source='cpp/train_ae_from_scratch.cpp',
                           includes='./cpp . ../../',
                           uselib=bld.env['magnum_libs']
                                  + 'ROBOTDART ROBOT_DART TBB BOOST EIGEN PTHREAD MPI'
                                  + ' DART DART_GRAPHIC'
                                  + ' PYTORCH LIBFASTSIM SDL ROBOX2D MAGNUMTEXT CUDA',
                           use='dbg_aurora',
                           target='train_ae_from_scratch',
                           # Taking the set of chosen experiments to avoid any duplicate
                           variants=[maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS[2].get_str_variables_run_experiment()],
                           )

