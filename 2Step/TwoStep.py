import numpy as np

class TwoStep:
    def __init__(self, filename, preClass=False, sep='\t', header=True, ignorList=[]):
        self.pClass = preClass
        self.ig = ignorList
        self.sep = sep
        self._symbolvar(filename, header)
        self.CCHash = {}
        self.diag = {}
        self.genMatrix(filename, header)
        self.CCmatrix = {}
        self.baseATT = None
        ### Predefine the Num of First Step Subcluster, 1/3 sample ###
        if preClass: self.preClass = preClass
        else: self.preClass = self.mtrxRnum/3
        ##############################################################

        ### Phase 1 ###
        self.findBase()
        self.nom2val = dict(zip(self.mxOrder.keys(), [0]*len(self.mxOrder.keys())))

        ############################################################################
        ##                                                                        ##
        ## Phase 1 is End, and self.CCmatrix is The Simliar Matrix of Nominal Var ##
        ##                                                                        ##
        ############################################################################

        ### Phase 2, update matrix with new numerical var ###
        nominal_useless = []
        if self.ig: nominal_useless = map(lambda x: self.nominalArr.index(x), self.ig)
        self.baseGroup = self._baseGp()
        self.baseNum = min(map(lambda x: self.cmptGV(x), range(self.matrixNum.shape[1])), key=lambda n: n[1])[0]
        self.nom2val.update(dict(map(lambda x: [x, np.mean(self.matrixNum[self.baseGroup[x], self.baseNum])], self.baseGroup)))
        for mm in self.mxOrder:
            if mm in self.baseGroup.keys(): continue
            self.nom2val[mm] = sum(map(lambda x: self._nomSimilar(self.mxOrder[x], self.mxOrder[mm])*self.nom2val[x], self.baseGroup.keys()))

        for mm in self.nom2val:
            self.matrixNom[self.matrixNom == mm] = self.nom2val[mm]

        ## New Matrix for Clustering ##
        self.clustMtx = np.hstack((self.matrix, self.matrixNom[:, filter(lambda x: x not in nominal_useless, range(len(self.nominalArr)))]))

        ###################################################################
        ##                                                               ##
        ##      Phase 2 end, now all nominal var has a num mapping       ##
        ##                                                               ##
        ###################################################################


        
        

    def _nomSimilar(self, ord1, ord2):
        m12 = self.CCmatrix[ord1, ord2] or self.CCmatrix[ord2, ord1]
        m11 = self.CCmatrix[ord1, ord1]
        m22 = self.CCmatrix[ord2, ord2]
        return m12/(m11+m22-m12)


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

    def _coOccur(self, tArray1, tArray2):
        for i in range(tArray1.shape[0]):
            if self.CCHash.has_key('\t'.join([tArray1[i], tArray2[i]])):
                self.CCHash['\t'.join([tArray1[i], tArray2[i]])] += 1
            else:
                self.CCHash['\t'.join([tArray1[i], tArray2[i]])] = 0

    def findBase(self):
        ## Order for all Feature Value ##
        m = 0
        nominalArr = filter(lambda x: x not in self.ig, self.nominalArr)
        ## [value, key] ##
        self.baseATT = max(map(lambda x: [len(np.unique(self.matrix[:, x])), x], nominalArr), key=lambda x: x[0])

        ##################
        tmpmtx = self.matrix[:, self.baseATT[1]]
        self.mxOrder = dict([[ele, n] for n, ele in enumerate(np.unique(tmpmtx))])
        m = n+1
        nominalArr.remove(self.baseATT[1])
        for i in nominalArr:
            self.mxOrder.update(dict([[ele, n+m] for n, ele in enumerate(np.unique(self.matrix[:, i]))]))
            m += len(np.unique(self.matrix[:,i]))
            self._coOccur(tmpmtx, self.matrix[:, i])

        self.CCmatrix = np.zeros((len(self.mxOrder.keys()), len(self.mxOrder.keys())))
        ## Compute Diagonal ##
        nominalArr.append(self.baseATT[1])
        for i in nominalArr:
            tmpatt = np.unique(self.matrix[:, i])
            tmpval = map(lambda x: list(self.matrix[:, i]).count(x), np.unique(self.matrix[:, i]))
            for n in range(len(tmpatt)):
                self.CCmatrix[self.mxOrder[tmpatt[n]], self.mxOrder[tmpatt[n]]] = float(tmpval[n])

        for tkey in self.CCHash:
            tkeys = tkey.split('\t')
            self.CCmatrix[self.mxOrder[tkeys[0]], self.mxOrder[tkeys[1]]] = self.CCHash[tkey]

    def _similarCOEF(self, nomVar1, nomVar2):
        ## D(a,c) = M(a, c) /(M(a) + M(c) - M(a,c)) ##
        return self.CCmatrix[self.mxOrder[nomVar1], self.mxOrder[nomVar1]]/(self.CCmatrix[self.mxOrder[nomVar1], self.mxOrder[nomVar1]] + self.CCmatrix[self.mxOrder[nomVar2], self.mxOrder[nomVar2]] - self.CCmatrix[self.mxOrder[nomVar1], self.mxOrder[nomVar2]])

    def _baseGp(self):
        a = {}
        for n, ele in sorted(enumerate(self.matrix[:, self.baseATT[1]]), key=lambda x: x[1]):
            if a.has_key(ele):
                a[ele].append(n)
            else:
                a[ele] = [n]
        return a
                
    def cmptGV(self, cIndice):
        m = 0
        for i in self.baseGroup:
            m += np.var(self.matrixNum[self.baseGroup[i], cIndice])*len(self.baseGroup[i])

        return [cIndice, m]
