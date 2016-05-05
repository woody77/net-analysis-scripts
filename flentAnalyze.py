"""
This generates graphs from a series of results of flent runs, using the data that's exported in gzip'd JSON format

 The general usage is:

 python flentAnalyze.py -i "dataset1 name" "dataset1 file" [-i "dataset2 name" "dataset2 file"]

 The dataset names are used as display labels instead of the filenames in all of the graphs.
"""

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import json
from pprint import pprint

"""
First snag a whole bunch of arguments, then get on with the actual data processing
"""


class NameValuePairArgList(argparse.Action):
    """
    This is for arguments that are lists of name=value pairs.  Multiple uses of the same destination with new pairs
    just gets added to the same list (not a list of lists)
    """

    def __call__(self, parser, namespace, values, option_string=None):
        for pair in values:
            # pprint(pair)
            n, v = pair.split('=')
            if getattr(namespace, self.dest) is None:
                # need to create the list if it doesn't exist
                setattr(namespace, self.dest, [[n, v]])
            else:
                # otherwise just add to it
                getattr(namespace, self.dest).append([n, v])


class NameValuePairArgDict(argparse.Action):
    """
    This is for lists of name=value pairs where the desire is for a dictionary of name=value pairs (vs a list).
    Multiple uses of the same destination with more pairs just gets added to the same dictionary, possibly overwriting
    previous keys if a key is duplicated (no warnings).
    """

    def process_value(self, raw_value):
        return raw_value

    def __call__(self, parser, namespace, values, option_string=None):
        for pair in values:
            # pprint(pair)
            n, v = pair.split('=')
            v = self.process_value(v)
            if getattr(namespace, self.dest) is None:
                setattr(namespace, self.dest, {n: v})
            else:
                getattr(namespace, self.dest)[n] = v


class NameJSONPairArgDict(NameValuePairArgDict):
    """
    Same as its parent class, except the values are structured data, parsed from JSON format (useful for dealing with
    things that need to become lists/tuples
    """

    def process_value(self, raw_value):
        return json.loads(raw_value)


arg_parser = argparse.ArgumentParser(description='Process IPerf3 JSON output into graphs')

arg_parser.add_argument('--output_base', nargs='?',
                        help="base filename for graphs to be saved to "
                             "(defaults to first input filename, without the last extension")
arg_parser.add_argument('--data', nargs='+', action=NameValuePairArgList, required=True,
                        help='<label>=<filename> pairs of labels and flent JSON files')
arg_parser.add_argument('--fig', nargs='*', action=NameJSONPairArgDict,
                        help="name=value pairs of matplotlib figure options")

arg_parser.add_argument('--graphs', nargs='*',
                        choices=["bw", "stats", "pps", "retransmits", "cwnd", "cwnd_pkts", "rtt", "cdf"],
                        default=["bw", "stats", "cwnd", "cdf"])

args = arg_parser.parse_args()

if args.output_base is None:
    first_filename = args.data[0][1]
    try:
        args.output_base = first_filename[:first_filename.rindex('.')]
    except ValueError:
        args.output_base = first_filename

pprint(args)

"""
These are data-import utility methods.  Here to get our data loaded into a useful form.
"""


def get_throughput_json(filename):
    """
    Loads a JSON file into memory
    :param filename: JSON file to parse
    :return: parsed data
    """
    with open(filename) as json_data:
        d = json.load(json_data)
        json_data.close()
        return d


class Sample(object):
    """
    Holds the sample data as an object, which is easier to deal with than a dict
    """

    def __init__(self, bw, seconds):
        self.seconds = seconds
        self.bw = bw

def sum_samples(*samples):
    total_bw = sum([s.bw for s in samples if s is not None])
    mean_time = sum([s.seconds for s in samples if s is not None])/len(samples)
    return Sample(total_bw, mean_time)


def extract_stream_data(json_data, print_samples=False):
    """
    list extract_stream_data(Object, str)

    Takes the parsed JSON object for a dataset and converts it into a list of sample objects

    :param json_data: parsed JSON data from IPerf
    :param print_samples: debug print out the samples after processing
    :return: list of Sample objects
    """
    # pprint(json['intervals'][0])
    # print ""
    # print "Intervals:"
    parsed_data = [Sample(raw_sample['val'], raw_sample['dur']) for raw_sample in json_data]

    if print_samples:
        for sample in parsed_data:
            print "    seconds: {:1.3f}, bw: {:3.1f}".format(sample.seconds, sample.bw)
    return parsed_data


"""
And now we get to the useful bit, that actually does stuff, like the reading in of data and parsing of it.

First, load each of the datasets into a name, data tuple
"""
color_map = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
color_id = 0
datasets = []

for dataset_name, dataset_file in args.data:
    # dataset_name = dataset[0]
    # dataset_file = dataset[1]
    print "\nProcessing:", dataset_name, "from", dataset_file
    json_data = get_throughput_json(dataset_file)

    extracted_streams=[]

    for stream_name, stream_data in [(stream_name, stream_data) for (stream_name,stream_data) in json_data['raw_values'].iteritems() if stream_name.startswith("TCP")]:
        extracted_data = extract_stream_data(stream_data)
        print stream_name, len(extracted_data), "samples"
        extracted_streams.append(extracted_data)

    samples_to_use = min([len(stream) for stream in extracted_streams])

    summation = map(sum_samples, *extracted_streams)
    datasets.append((dataset_name, summation, color_map[color_id]))
    color_id = (color_id + 1) % len(color_map)

"""
Next, we need to determine what the graph constraints are.  This is based on the size of the datasets.
"""

packet_mtu = 1500
xmax = max([len(data) for name, data, color in datasets])
xticks = range(0, xmax, xmax / 20)  # make this smarter later
# bw_ymax = 1e9
# bw_yticks = (range(0, int(1e9), int(1e8)), range(0, 1000, 100))
fig_params = args.fig


def generate_limit_and_ticks(max_value):
    """
    Estimates a max value that is reasonably larger than the max value (to provide a graph).  Then returns a limit and
    set of tick points that are on even powers (ticks are an order of magnitude smaller).

    :type (float) -> float, [float]
    :param max_value: the maximum value being plotted
    :return: a reasonable, even, value to use as the graph max and a list of the ticks that's reasonable
    """
    print "max value:",max_value
    order = math.log10(max_value)
    multiple, limit_power = math.modf(order)
    multiple = math.pow(10, abs(multiple))

    print "base power and extension:", limit_power, multiple

    # within an order of magnitude, there are three ranges that are useful to look at:
    if multiple > 5:
        # over 5.0x10^x, and we care about that decade as a whole
        limit_power += 1
        num = 1
        extension = 0
    elif multiple > 2:
        # over 2.0*10^X, and we care about the first half of the decade (0-5.0)*10^x
        num = 5
        extension = 0
    else:
        # below 2.0*10^X, we really care about 20*10^(x-1)

        extension, num = math.modf(multiple)
        print "multiple num and extension:", num, extension

        # And the same 5/2/10 breakout repeats at this level, but adds to 10 to give more buckets over (10+n)*10^(x-1)
        num = 1
        if extension > 0.5:
            num += 1
            extension = 0
        elif extension > 0.2:
            extension = 0.5
        elif extension > 0.1:
            extension = 0.2
        elif extension > .001:
            extension = 0.1
        else:
            extension = 0

    print "new power, number, extension:", limit_power, num, extension

    max_value = math.pow(10, limit_power)* (num+extension)
    print "new max: (number+extension)*10^power:", max_value

    if 3 <= num < 10:
        steps = num
    else:
        steps = max_value / math.pow(10, limit_power-1)

    print "steps:",steps

    ticks = np.linspace(0, max_value, steps+1, endpoint=True)
    print "steps:", ticks

    return max_value, ticks


def generate_limit_and_ticks_K(max_value):
    """
    Estimates a max value that is reasonably larger than the max value (to provide a graph).  Then returns a limit and
    set of tick points that are on even powers and multiples of 1024 (as opposed to 10)

    :type (float) -> float, [float]
    :param max_value: the maximum value being plotted
    :return: a reasonable, even, value to use as the graph max and a list of the ticks that's reasonable
    """
    print "max value:",max_value
    order = math.log(max_value, 1024)
    multiple, limit_power = math.modf(order)
    multiple = math.pow(1024, multiple)

    print "base power and extension:", limit_power, multiple

    # here, we have a few more complications, because instead of decades we're looking at SI prefix jumps (millenniums)
    if multiple > 512:
        # again, 5s get pushed up to the next order
        limit_power += 1
        num = 1
        extension = 0
    elif multiple > 256:
        # twos get pushed to 512 (4*128)
        num = 4
        extension = 0
    else:

        # and the rest get sifted again, in numbers of 128
        extension, num = math.modf(multiple/128)
        print "multiple num and extension:", num, extension

        num = 8
        if extension > 0.5:
            num += 8
            extension = 0
        elif extension > 0.2:
            extension = 0.5
        elif extension > 0.1:
            extension = 0.2
        elif extension > .001:
            extension = 0.1
        else:
            extension = 0

    print "new power, number, extension:", limit_power, num, extension

    max_value = math.pow(10, limit_power)* (num+extension)
    print "new max: (number+extension)*10^power:", max_value

    if 2 <= num < 10:
        steps = num
    else:
        steps = max_value / math.pow(10, limit_power-1)

    print "steps:",steps

    ticks = np.linspace(0, max_value, steps+1, endpoint=True)
    print "steps:", ticks

    return max_value, ticks


def build_graph(title):
    # :type (str) -> None
    print '\nBuilding graph: ' + title
    plt.figure(**fig_params)
    plt.title(title)
    plt.grid(True)


def build_time_graph():
    plt.xlim(0, xmax)
    plt.xticks(xticks)
    plt.xlabel('seconds')


def build_plots(datasets):
    max_value = None
    for name, data, plot_color in datasets:
        max_value = max(max_value, *data)
        plt.plot(data, label=name, color=plot_color)
    return max_value


def build_bw_graph(datasets):
    max_value = build_plots(datasets)
    ylim, ticks = generate_limit_and_ticks(max_value)
    plt.ylim(0, ylim)
    plt.yticks(ticks)
    plt.ylabel('Mbps')


def build_count_graph(datasets, counts_of=None):
    max_value = build_plots(datasets)
    ylim, ticks = generate_limit_and_ticks(max_value)
    plt.ylim(0, ylim)
    plt.yticks(ticks)
    if counts_of is not None:
        plt.ylabel(counts_of)


def build_legend(loc=None):
    if loc is None:
        plt.legend()
    else:
        plt.legend(loc=loc)


def save_graph(label):
    filename = args.output_base + "." + label + ".png"
    print "Saving to < "+filename+" >..."
    plt.savefig(filename)


"""
And the real purpose of all of this, to make a bunch of graphs!!!

"""
pprint(args.graphs)
print "\n Creating graphs: "+", ".join(args.graphs)

"""
This is a graph of all of the bandwidths, over time.
"""
if "bw" in args.graphs:
    build_graph("Throughput Bandwidth")
    build_time_graph()
    build_bw_graph([(name, [s.bw for s in data], color) for name, data, color in datasets])
    build_legend(loc=4)  # lower right, which works better for this particular one
    save_graph("bw")

"""
This is a graph of a bunch of descriptive statistics of the bandwidth, overlaid on the bandwidth
"""

if "stats" in args.graphs:
    build_graph("Throughput Bandwidth Statistics")
    build_time_graph()
    max_value = None
    for name, data, color in datasets:
        bw = np.array([s.bw for s in data])

        max_value = max(max_value, bw.max())

        mean = bw.mean()
        sd = bw.std()
        sem = sd / math.sqrt(len(bw))
        print name,mean,sem

        plt.fill([0, xmax, xmax, 0], [mean - sem, mean - sem, mean + sem, mean + sem],
                 alpha=0.3, color=color)
        plt.hlines(bw.mean() - 3 * bw.std(), 0, xmax, linestyle=':', color=color)
        plt.hlines(bw.mean() - 2 * bw.std(), 0, xmax, linestyle='-.', color=color)
        plt.hlines(bw.mean() - bw.std(), 0, xmax, linestyle='--', color=color)
        plt.hlines(bw.mean(), 0, xmax, color=color)
        plt.plot(bw, label=name, color=color)

    ylim, ticks = generate_limit_and_ticks(max_value)
    plt.ylim(0, ylim)
    plt.yticks(ticks)
    build_legend(loc=4)
    save_graph("stats")

"""
This is a graph of the estimate of the number of packets per second being sent.
"""

if "pps" in args.graphs:
    build_graph("Packets per second")
    build_time_graph()
    build_count_graph([(name, [s.bw / (8*packet_mtu) for s in data], color) for name, data, color in datasets], counts_of="packets/second")
    build_legend(loc=4)  # lower right, which works better for this particular one
    save_graph("pps")

"""
This is a CDF of the bandwidth, which is very useful for comparing the overall response of multiple versions/setups
"""

if "cdf" in args.graphs:
    build_graph("Cumulative Distribution of Throughput")

    max_value,ticks = generate_limit_and_ticks(max([s.bw for s in data for data in name,data,color in datasets]))

    hist_xpoints = np.linspace(0, max_value, 1001, endpoint=True)
    hist_xticks = ticks
    hist_xlabels = ["{:4.0f}".format(t) for t in hist_xticks]

    for name, data, color in datasets:
        plt.plot(hist_xpoints[:-1],
                 stats.cumfreq([s.bw for s in data], hist_xpoints, (0, 1e9))[0] / len(data),
                 label=name,
                 color=color)

    plt.xticks(hist_xticks, hist_xlabels)
    hist_yticks = np.arange(0, 101, 10) / 100.0
    plt.yticks(hist_yticks, ["{:2.0%}".format(float(t)) for t in hist_yticks])
    plt.xlabel("Mbps")
    plt.ylabel("percentile")

    build_legend(loc=4)
    save_graph("cdf")
