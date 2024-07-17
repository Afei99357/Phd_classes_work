# Eric	BINF6203_001	SP21	3/8/2021
import argparse

# infile = open(input("what is the name of the input file?\n>>>"), 'r')
# outfile = open(input("What would you like to name your output file?\n>>>"), 'w')

parser = argparse.ArgumentParser(description="Reads factors for each sample and writes them into CSV file")
parser.add_argument('--input_file', help='files contains chromosome SNPs information',
                    required=True)
parser.add_argument('--output_file', help='number of SNPs for each chromosome', required=True)

args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file
n1 = n2 = n3 = n4 = n5 = n6 = n7 = n8 = n9 = n10 = n11 = n12 = \
    n13 = n14 = n15 = n16 = n17 = n18 = n19 = n20 = n21 = n22 = 0
with open(input_file, mode='r') as vcf:
    for line in vcf:
        line = line.strip("\n")
        if line.startswith("chr1"):
            n1 += 1
        if line.startswith("chr2"):
            n2 += 1
        if line.startswith("chr3"):
            n3 += 1
        if line.startswith("chr4"):
            n4 += 1
        if line.startswith("chr5"):
            n5 += 1
        if line.startswith("chr6"):
            n6 += 1
        if line.startswith("chr7"):
            n7 += 1
        if line.startswith("chr8"):
            n8 += 1
        if line.startswith("chr9"):
            n9 += 1
        if line.startswith("chr10"):
            n10 += 1
        if line.startswith("chr11"):
            n11 += 1
        if line.startswith("chr12"):
            n12 += 1
        if line.startswith("chr13"):
            n13 += 1
        if line.startswith("chr14"):
            n14 += 1
        if line.startswith("chr15"):
            n15 += 1
        if line.startswith("chr16"):
            n16 += 1
        if line.startswith("chr17"):
            n17 += 1
        if line.startswith("chr18"):
            n18 += 1
        if line.startswith("chr19"):
            n19 += 1
        if line.startswith("chr20"):
            n20 += 1
        if line.startswith("chr21"):
            n21 += 1
        if line.startswith("chr22"):
            n22 += 1

with open(output_file, mode='w') as output:
    output.write('chrom1' + '\t' + str(n1) + '\n' +
                 'chrom2' + '\t' + str(n2) + '\n' +
                 'chrom3' + '\t' + str(n3) + '\n' +
                 'chrom4' + '\t' + str(n4) + '\n' +
                 'chrom5' + '\t' + str(n5) + '\n' +
                 'chrom6' + '\t' + str(n6) + '\n' +
                 'chrom7' + '\t' + str(n7) + '\n' +
                 'chrom8' + '\t' + str(n8) + '\n' +
                 'chrom9' + '\t' + str(n9) + '\n' +
                 'chrom10' + '\t' + str(n10) + '\n' +
                 'chrom11' + '\t' + str(n11) + '\n' +
                 'chrom12' + '\t' + str(n12) + '\n' +
                 'chrom13' + '\t' + str(n13) + '\n' +
                 'chrom14' + '\t' + str(n14) + '\n' +
                 'chrom15' + '\t' + str(n15) + '\n' +
                 'chrom16' + '\t' + str(n16) + '\n' +
                 'chrom17' + '\t' + str(n17) + '\n' +
                 'chrom18' + '\t' + str(n18) + '\n' +
                 'chrom19' + '\t' + str(n19) + '\n' +
                 'chrom20' + '\t' + str(n20) + '\n' +
                 'chrom21' + '\t' + str(n21) + '\n' +
                 'chrom22' + '\t' + str(n22))
