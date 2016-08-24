
import sys
import random
out= open(sys.argv[1],'w')
num= int(sys.argv[2])
nOut=int(sys.argv[3])

for i in range(num):
    while 1:
        length = random.randint(2,nOut/2)
        if nOut%length==0:
            break;
    inStr=''
    targetStr=''
    targetPos=0
    targetPortion = nOut/length 
    randChar=''
    lastRC=''
    for ii in range(length):
        while randChar==lastRC:
            randChar=str(unichr(random.randint(ord('a'),ord('d'))))
        inStr+=randChar
        for iii in range(0,targetPortion-1):
            targetStr+=randChar
        targetStr+=randChar.upper()
        lastRC=randChar
    for ii in range(len(targetStr),nOut):
        targetStr+=randChar
    out.write(inStr+' '+targetStr+'\n')

out.close()
        
