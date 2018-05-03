"""@package test
Test harness for gtracer
"""

from math import floor
import sys
import subprocess
import multiprocessing
import csv_mt

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

CSI         = "\x1b["     # ANSI CSI escape sequence
CSI_UP      = CSI + "1A"  # move cursor up
CSI_CLEARLN = CSI + "2K"  # clear the line

def main():
    test_accel()
    #test_threading()

def test_accel(path="accel.png"):
    # base test parameters
    base_cmd = "./tracer --stats --gpu -n 10 -w 1024 -h 1024"
    obj_paths = ["obj/test1002.obj", "obj/test1k2.obj", "obj/test10k2.obj"]
    obj_names = ["100 tris", "1,000 tris", "10,000 tris"]
    obj_settings = ["-S1", "-S1", "-S1"]

    # compile commands for each test type
    base_obj_cmds = ["{} {} {}".format(base_cmd, obj, sobj) 
                     for obj, sobj in zip(obj_paths, obj_settings)]
    accel_cmds = list(base_obj_cmds)
    brute_cmds = [base + " --daccel" for base in base_obj_cmds]
    zipped = zip(brute_cmds, accel_cmds)

    # run tests and collect timing data
    brute_times = []
    accel_times = []
    for brute_cmd, accel_cmd in zipped:
        brute_times.append(test_time(brute_cmd))
        accel_times.append(test_time(accel_cmd))

    print("Brute-force times: ", brute_times)
    print("Accelerated times: ", accel_times)
    
    # compute pthread vs cpu and gpu vs cpu speedups
    speedups = [brute / accel 
                        for brute, accel in zip(brute_times, accel_times)]

    # plot test results
    cmap = plt.get_cmap("summer")
    n_points = len(speedups)
    colors = [cmap(i / (n_points) + 0.5 / n_points) for i in range(n_points)]
    
    plot_bars(
        barGroups = [speedups],
        barNames = obj_names,
        groupNames = ["Uniform Grid Acceleration"],
        ylabel = "Speedup factor",
        title = "Speedup vs. Brute-Force",
        legendTitle = "Input file",
        colors = colors,
        chart_width = 0.8)
    # plt.show()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print("writing plot to {}".format(path))

def test_threading(path="test.png"):
    # base test parameters
    base_cmd = "./tracer --stats -n 10 -w 1024 -h 1024"
    obj_paths = ["obj/test1002.obj", "obj/test1k2.obj", "obj/test10k2.obj", "obj/test100k2.obj"]
    obj_names = ["100 tris", "1,000 tris", "10,000 tris", "100,000 tris"]
    obj_settings = ["-S1", "-S1", "-S1", "-S1"]

    # compile commands for each test type
    base_obj_cmds = ["{} {} {}".format(base_cmd, obj, sobj) 
                     for obj, sobj in zip(obj_paths, obj_settings)]
    cpu_cmds = [base + " -t1" for base in base_obj_cmds]
    pthread_cmds = list(base_obj_cmds)
    gpu_cmds = [base + " -g" for base in base_obj_cmds]
    zipped = zip(cpu_cmds, pthread_cmds, gpu_cmds)

    # run tests and collect timing data
    cpu_times = []
    pthread_times = []
    gpu_times = []
    for cpu_cmd, pthread_cmd, gpu_cmd in zipped:
        cpu_times.append(test_time(cpu_cmd))
        pthread_times.append(test_time(pthread_cmd))
        gpu_times.append(test_time(gpu_cmd))

    print("CPU times: ", cpu_times)
    print("pthread times: ", pthread_times)
    print("GPU times:", gpu_times)
    
    # compute pthread vs cpu and gpu vs cpu speedups
    speedups_pthread = [cpu / pthread 
                        for cpu, pthread in zip(cpu_times, pthread_times)]
    speedups_gpu = [cpu / gpu for cpu, gpu in zip(cpu_times, gpu_times)]

    # plot test results
    cmap = plt.get_cmap("summer")
    n_points = len(speedups_gpu)
    colors = [cmap(i / (n_points) + 0.5 / n_points) for i in range(n_points)]
    
    plot_bars(
        barGroups = [speedups_pthread, speedups_gpu],
        barNames = obj_names,
        groupNames = ["CPU x8", "GPU"],
        ylabel = "Speedup factor",
        title = "Speedup vs. Single-threaded",
        legendTitle = "Input file",
        colors = colors,
        chart_width = 0.8)
    # plt.show()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print("writing plot to {}".format(path))

def plot_bars(barGroups, barNames, groupNames, colors, ylabel="", title="", legendTitle="", width=0.8, chart_width=0.8):
    """Plot a grouped bar chart
    barGroups  - list of groups, where each group is a list of bar heights
    barNames   - tuple containing the name of each bar within any group
    groupNames - tuple containing the name of each group
    colors     - list containing the color for each bar within a group
    ylabel     - label for the y-axis
    title      - title
    """
    fig, ax = plt.subplots()
    offset = lambda items, off: [x + off for x in items]

    maxlen = max(len(group) for group in barGroups)
    xvals = range(len(barGroups))
    
    for i, bars in enumerate(zip(*barGroups)):
        plt.bar(
            x = offset(xvals, i * width/maxlen), 
            height = bars, 
            width = width/maxlen, 
            color=colors[i])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(offset(xvals, width / 2 - width / maxlen / 2))
    ax.set_xticklabels(groupNames)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * chart_width, box.height])

    # Put a legend to the right of the current axis
    ax.legend(barNames, title=legendTitle, loc="upper left", bbox_to_anchor=(1, 1))

def test_time(cmd, samples=3, warmup=1):
    """Run shell command cmd with a series of arguments.
    cmd      - a shell command to run.
    samples  - how many times to run the test and average timing data.
    warmup   - extra samples that will be discarded to "warm up" the system.
    """
    # do testing
    print()
    avg_time = 0
    for s in range(samples + warmup):
        # report progress
        progress = s / (samples + warmup)
        print(CSI_UP + CSI_CLEARLN + "Testing [{}%]".format(floor(progress * 100)))

        while True:
            try:
                output = shell(cmd)                           # run command
                break
            except:
                pass
        tables = csv_mt.read_string(output, parse_float=True) # parse its output
        time = tables["statistics"]["total_time_ms"][0]       # get its timing data

        # skip a few runs to let the system "warm up"
        if s >= warmup:
            avg_time += time / samples # compute average execution time

    # log the average time for this test case
    return avg_time

def fmt_time(t):
    return "{:.4f} sec".format(t)

def shell(cmd):
    """Return the result of running a shell command.
    cmd - a string representing the command to run.
    """
    return subprocess.check_output(cmd, shell=True).decode("utf-8")

main()
