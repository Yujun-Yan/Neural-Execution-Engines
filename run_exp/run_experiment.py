from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import subprocess
import argparse
import time
from run_shortest_path import run_dist
from run_sel_sort import run_sel
from run_merge_sort import run_merge
from run_addition import run_add
from run_addition_holdout import run_add_hold
from run_mul import run_mul
from run_prim import run_prim
from run_min_graph import run_min_graph

############ This file is a wrap-up for different tasks and distribute parameters accordingly ##############
parser = argparse.ArgumentParser()
parser.add_argument('-T', "--Task", choices=["add", "addhold", "mul", "merge", "sel", "dist", "mst", "predist", "premst"], required=True, help='choose from "add", "addhold", "mul", "merge", "sel", "dist","mst", "predist", "premst"')
parser.add_argument('-R', "--Reload", type=str, nargs='+', default=["No"], help='Paths (w.r.t current folder) to reload the model')
parser.add_argument('-H', "--Holdout", choices=list(range(8, 193)), type=int, default=166, help='Number of holdout: 8<=N<=192 ')


args = parser.parse_args() 

home_dir = "./"
file_name = args.Task
current_time = time.strftime("%d_%H_%M_%S", time.localtime(time.time()))
current_dir = home_dir + file_name + "_" + current_time

subprocess.call("mkdir {}".format(current_dir), shell=True)
current_dir += '/'
if args.Reload[0] == "No":
    reload_from_dir = False
else:
    reload_from_dir = True

################################ Reloading...##########################################
if (args.Task == 'dist') and (len(args.Reload)!=2):
    raise ValueError('Two paths of directories are needed for reloading Dijkstra shortest path')
if (args.Task == 'mst') and (len(args.Reload)!=2):
    raise ValueError('Two paths of directories are needed for reloading Prim algorithm')
if (len(args.Reload)!=1) and (reload_from_dir) and (args.Task != 'dist') and (args.Task != 'mst'):
    raise ValueError('Only one path is needed for reloading the model')  

if args.Task == 'dist':
    reload_dir_1 = home_dir + args.Reload[0] + "/"
    reload_dir_2 = home_dir + args.Reload[1] + "/"
    checkpoint_path_1 = current_dir + "model_1"
    checkpoint_path_2_data = current_dir + "model_2_data"
    checkpoint_path_2_msk = current_dir + "model_2_msk"
    subprocess.call("cp -r {}model_1 {}".format(reload_dir_1, current_dir), shell=True)
    subprocess.call("cp -r {}model_2_data {}".format(reload_dir_2, current_dir), shell=True)
    subprocess.call("cp -r {}model_2_msk {}".format(reload_dir_2, current_dir), shell=True)
    run_dist(current_dir, reload_dir_1, reload_dir_2, checkpoint_path_1, checkpoint_path_2_data, checkpoint_path_2_msk)
elif args.Task == 'mst':
    reload_dir_data = home_dir + args.Reload[0] + "/"
    reload_dir_msk = home_dir + args.Reload[1] + "/"
    checkpoint_path_data_1 = current_dir + "data_1"
    checkpoint_path_data_2 = current_dir + "data_2"
    checkpoint_path_msk = current_dir + "mask"
    subprocess.call("cp -r {}model_2_data {}/data_1".format(reload_dir_data, current_dir), shell=True)
    subprocess.call("cp -r {}model_2_data {}/data_2".format(reload_dir_data, current_dir), shell=True)
    subprocess.call("cp -r {}model_2_msk {}/mask".format(reload_dir_msk, current_dir), shell=True)
    run_prim(current_dir, reload_dir_data, reload_dir_msk, checkpoint_path_data_1, checkpoint_path_data_2, checkpoint_path_msk)
elif (args.Task == 'merge'):
    reload_dir = home_dir + args.Reload[0]+ "/"
    checkpoint_path_data = current_dir + "model_data"
    checkpoint_path_msk = current_dir + "model_msk"
    if reload_from_dir:
        subprocess.call("cp -r {}model_data {}".format(reload_dir, current_dir), shell=True)
        subprocess.call("cp -r {}model_msk {}".format(reload_dir, current_dir), shell=True)
        subprocess.call("cp {}*.npy {}".format(reload_dir, current_dir), shell=True)
    else:
        subprocess.call("mkdir {}".format(checkpoint_path_data), shell=True)
        subprocess.call("mkdir {}".format(checkpoint_path_msk), shell=True)
    run_merge(current_dir, reload_from_dir, reload_dir, checkpoint_path_data, checkpoint_path_msk)
elif (args.Task == "sel"):
    reload_dir = home_dir + args.Reload[0] + "/"
    checkpoint_path_data = current_dir + "model_2_data"
    checkpoint_path_msk = current_dir + "model_2_msk"
    if reload_from_dir:
        subprocess.call("cp -r {}model_2_data {}".format(reload_dir, current_dir), shell=True)
        subprocess.call("cp -r {}model_2_msk {}".format(reload_dir, current_dir), shell=True)
        subprocess.call("cp {}*.npy {}".format(reload_dir, current_dir), shell=True)
    else:
        subprocess.call("mkdir {}".format(checkpoint_path_data), shell=True)
        subprocess.call("mkdir {}".format(checkpoint_path_msk), shell=True)
    run_sel(current_dir, reload_from_dir, reload_dir, checkpoint_path_data, checkpoint_path_msk)
elif (args.Task == "predist") or (args.Task == "premst"):
    if args.Task == "predist":
        mst = False
    else:
        mst = True
    reload_dir = home_dir + args.Reload[0] + "/"
    checkpoint_path_data = current_dir + "model_2_data"
    checkpoint_path_msk = current_dir + "model_2_msk"
    if reload_from_dir:
        subprocess.call("cp -r {}model_2_data {}".format(reload_dir, current_dir), shell=True)
        subprocess.call("cp -r {}model_2_msk {}".format(reload_dir, current_dir), shell=True)
        subprocess.call("cp {}*.npy {}".format(reload_dir, current_dir), shell=True)
    else:
        subprocess.call("mkdir {}".format(checkpoint_path_data), shell=True)
        subprocess.call("mkdir {}".format(checkpoint_path_msk), shell=True)
    run_min_graph(current_dir, reload_from_dir, reload_dir, checkpoint_path_data, checkpoint_path_msk, mst)
else:
    reload_dir = home_dir + args.Reload[0]+ "/"
    checkpoint_path = current_dir + "model_1"
    if reload_from_dir:
        subprocess.call("cp -r {}model_1 {}".format(reload_dir, current_dir), shell=True)
        subprocess.call("cp {}*.npy {}".format(reload_dir, current_dir), shell=True)
    else: 
        subprocess.call("mkdir {}".format(checkpoint_path), shell=True)
    if args.Task == "add":
        run_add(current_dir, reload_from_dir, reload_dir, checkpoint_path)
    elif args.Task == "addhold":
        run_add_hold(current_dir, reload_from_dir, reload_dir, checkpoint_path, args.Holdout)
    else:
        run_mul(current_dir, reload_from_dir, reload_dir, checkpoint_path)