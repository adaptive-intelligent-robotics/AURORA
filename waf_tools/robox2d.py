#! /usr/bin/env python
# encoding: utf-8
# Antoine Cully - 2019 from Konstantinos Chatzilygeroudis - 2015

"""
Quick n dirty robox2d detection
"""

import os
from waflib.Configure import conf


def options(opt):
    opt.add_option('--robox2d', type='string', help='path to robox2d', dest='robox2d')

@conf
def check_robox2d(conf):
  def get_directory(filename, dirs):
    res = conf.find_file(filename, dirs)
    return res[:-len(filename)-1]
  

  includes_check = ['/usr/local/include', '/usr/include']
  libs_check = ['/usr/local/lib', '/usr/lib']

  # OSX/Mac uses .dylib and GNU/Linux .so
  suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'

  # You can customize where you want to check
  # e.g. here we search also in a folder defined by an environmental variable
  if 'RESIBOTS_DIR' in os.environ:
    includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check
    libs_check = [os.environ['RESIBOTS_DIR'] + '/lib'] + libs_check

  if conf.options.robox2d:
    includes_check = [conf.options.robox2d + '/include']
    libs_check = [conf.options.robox2d + '/lib']

    try:
        conf.start_msg('Checking for robox2d includes')
        res = conf.find_file('robox2d/simu.hpp', includes_check)
        res = res and conf.find_file('robox2d/robot.hpp', includes_check)
        res = res and conf.find_file('robox2d/common.hpp', includes_check)
    
        include_robox2d = get_directory('robox2d/simu.hpp', includes_check)
        conf.env.INCLUDES_ROBOX2D = include_robox2d
        conf.end_msg('ok: ' +include_robox2d)
    except:
        conf.fatal('Includes not found')
        return

    try:
        conf.start_msg('Checking for robox2d libs') 
        res = conf.find_file('libRobox2d.' + suffix, libs_check)
        res = conf.find_file('libRobox2dMagnum.' + suffix, libs_check)
        conf.env.LIBPATH_ROBOX2D = get_directory('libRobox2d.' + suffix, libs_check)

        conf.env.LIB_ROBOX2D = ['Robox2d']
        conf.env.LIB_ROBOX2D.append('Robox2dMagnum')
        
        conf.end_msg('libs: ' + str(conf.env.LIB_ROBOX2D[:]))
    except:
        conf.fatal('Libs not found')
        return
  
  return 1


