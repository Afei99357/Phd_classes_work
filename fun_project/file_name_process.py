import os
import re

direct = "/Volumes/Eric Liao/电视剧/white_collar/"

for i in os.listdir(direct):
    x = re.search("White\.Collar\.S\d\dE\d\d", i)
    if x and (x.group()+'.mkv') != i and not i.startswith("."):
        # print("n")
        os.rename(direct + i, direct + x.group() + ".mkv")
    else:
        continue
