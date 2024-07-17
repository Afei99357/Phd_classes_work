# Kat Jens	BINF6203_001	SP21	2/25/2021

infile = open(input("what is the name of the input file?\n>>>"), 'r')
outfile = open(input("What would you like to name your output file?\n>>>"), 'w')


chrDict = {}
for line in infile:
	line = line.strip()
	if line.startswith("chr"):
		chrom = line
	else:
		count = line
		outfile.write(chrom +  '\t' +  count + '\n')
