import numpy as np
import pandas as pd

class TwoStep:
    def __init__(self, filename, preClass=False, sep='\t', header=True, ignorList=False):
        self.pClass = preClass
        self.ig = ignorList
        self.sep = sep
        self._symbolvar(filename, header)
        self.genMatrix(filename, header)
        ### Predefine the Num of First Step Subcluster, 1/3 sample ###
        if preClass: self.preClass = preClass
        else: self.preClass = self.mtrxRnum/3
        ##############################################################


    ### Gen Matrix and Trans Nominal Var to Numeric ###
    def genMatrix(self, filename, header):
        if header: srow = 1
        else: srow = 0
        self.matrix = np.loadtxt(filename, dtype=np.str, skiprows = srow, delimiter=self.sep)
        self.matrixNum = self.matrix[:, self.numArr].astype(np.float)
        self.matrixNumNam = map(lambda x:self.varname[x], self.numArr)
        self.matrixNomNam = map(lambda x:self.varname[x], self.nominalArr)
        self.matrixNom = self.matrix[:, self.nominalArr].astype(np.str)


    ######          Reallocate Var Type          ######
    def _symbolvar(self, filename, header):
        self.nominalArr = []
        self.varname = []
        self.mtrxRnum = 0
        f = open(filename)
        ### Header=True, ig 1st row ###
        #if self.ig: iglist = self.ig
        if header:
            lines = f.readline().strip().decode('utf8').split(self.sep)
            self.mtrxCnum = len(lines)
            #map(lambda x: del lines[x], iglist)
            #map(lambda y: self.varname.append(lines[y]), filter(lambda x: not x in self.ig, range(len(lines))))
            self.varname = lines
        ##############################
        for line in f:
            lines = line.strip().split(self.sep)
            self.mtrxRnum += 1
            for indexNum, ele in enumerate(lines):
                if ele.isdigit() or ele.replace('.', '', 1).isdigit():
                    continue
                else:
                    ### Nominal Var ###
                    if indexNum in self.nominalArr: continue
                    else: self.nominalArr.append(indexNum)
        self.numArr = filter(lambda x: not x in self.nominalArr, range(self.mtrxCnum))
    ##############    Matrix Generated    ###############

    #########        Nominal to Numeric       ########
    # To Cmpt 2nd step, we need keep original matrix #
    ##################################################
    def findBase(self):
        nominalArr = filter(lambda x: x not in self.ig, self.nominalArr)
        print map(lambda x: [len(np.unique(self.matrix[:, x])), x], nominalArr)
        ## [value, key] ##
        baseATT = max(map(lambda x: [len(np.unique(self.matrix[:, x])), x], nominalArr), key=lambda x: x[0])
        ##################
        tmpmtx = self.matrix[:, baseATT[1]]
        nominalArr.remove(baseATT[1])
        for i in nominalArr:
            self.coOccur(tmpmtx, self.matrix[:, i])
        self.mxOrder = dict([[ele, n] for n, ele in enumerate(np.unique(tmpmtx))])
        CCmatrix = np.zeros((len(np.unique(tmpmtx)), len(np.unique(tmpmtx))))
        print '-'*20
        print self.mxOrder
        return 

    def coOccur(self, tArray1, tArray2):
        tmpHash = {}
        for i in range(tArray1.shape[0]):
            if tmpHash.has_key('\t'.join([tArray1[i], tArray2[i]])):
                tmpHash['\t'.join([tArray1[i], tArray2[i]])] += 1
            else:
                tmpHash['\t'.join([tArray1[i], tArray2[i]])] = 0

        print tmpHash


        


        

    


