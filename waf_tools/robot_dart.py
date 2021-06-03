#! /usr/bin/env python
# encoding: utf-8
# Antoine Cully - 2019 from Konstantinos Chatzilygeroudis - 2015

"""
Quick n dirty robot_dart detection
"""

import os
from waflib.Configure import conf


def options(opt):
    opt.add_option('--robot_dart', type='string', help='path to robot_dart', dest='robot_dart')

@conf
def check_robot_dart(conf):
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

    if conf.options.robot_dart:
        includes_check = [conf.options.robot_dart + '/include']
        libs_check = [conf.options.robot_dart + '/lib']

    try:
        conf.start_msg('Checking for robot_dart includes')
        res = conf.find_file('robot_dart/robot_dart_simu.hpp', includes_check)
        res = res and conf.find_file('robot_dart/robot.hpp', includes_check)
        res = res and conf.find_file('robot_dart/utils.hpp', includes_check)

        include_robotdart = get_directory('robot_dart/robot_dart_simu.hpp', includes_check)
        conf.env.INCLUDES_ROBOTDART = include_robotdart
        conf.end_msg('ok: ' +include_robotdart)
    except:
        conf.fatal('Includes not found')
        return

    try:
        conf.start_msg('Checking for robot_dart libs')
        res = conf.find_file('libRobotDARTSimu.' + suffix, libs_check)
        res = conf.find_file('libRobotDARTMagnum.' + suffix, libs_check)


        conf.env.LIB_ROBOTDART = ['RobotDARTSimu']
        conf.env.LIB_ROBOTDART.append('RobotDARTMagnum')
        conf.end_msg('libs: ' + str(conf.env.LIB_ROBOTDART[:]))
    except:
        conf.fatal('Libs not found')
        return

    return 1

