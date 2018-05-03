"""@package csv_mt
csv_mt allows reading of a multi-table CSV format described as follows:

Table name
header 1,header 2,header 3
data1,data1,data1
data2,data2,data2
{blank line}

There may be as many tables as desired in a single input. Tables should be 
separated by a single blank line at the end of the table. The separator may 
be changed from a comma to any string by passing the `separator` parameter.

Calling read() on the sample input above would produce the following dict:
{
	"Table name": {
		"header 1": ["data1", "data2"],
		"header 2": ["data1", "data2"],
		"header 3": ["data1", "data2"]
	}
}
"""

def read_file(filename, **kwargs):
	"""Parse a file by name.
	filename - the file to read
	kwargs   - see read_table()
	"""
	with open(filename, "r") as file:
		return read(file.readlines(), **kwargs)

def read_string(string, **kwargs):
	"""Parse a string in csv_mt format.
	string - the string to parse
	kwargs - see read_table()
	"""
	return read(string.splitlines(), **kwargs)

def read(lines, **kwargs):
	"""Parse an iterable of lines in csv_mt format.
	lines  - the lines to parse (an iterable)
	kwargs - see read_table()
	"""
	lines_iter = iter(map(lambda line: line.rstrip(), lines)) # strip newlines
	tables = {name: _read_table(lines_iter, **kwargs) 
	          for name in lines_iter}                         # read each table name and table
	return tables

def _read_table(lines, separator=",", parse_float=False):
	"""Parse a single table from a set of lines.
	lines       - the lines to parse (an iterator)
	separator   - the data separator (default comma, could be tab, etc)
	parse_float - whether to parse data values as numbers
	"""
	# Parse headers
	headers = next(lines).split(separator)

	# Initialize table such that each header has an empty dataset
	table = {header: [] for header in headers}

	# Parse lines
	for line in lines:
		# Check for end of table
		if line == "":
			break

		values = line.split(separator)
		for i, value in enumerate(values):
			if parse_float:
				value = float(value)
			table[headers[i]] += [value] # append value to dataset for its header

	return table