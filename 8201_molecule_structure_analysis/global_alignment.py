from Bio import pairwise2
from Bio.pairwise2 import format_alignment

X = "CGCATG"
Y = "ACGAG"

alignments = pairwise2.align.globalms(X, Y, 1, 0, -0.5, 0)

for a in alignments:
    print(format_alignment(*a))