__author__ = 'kos'

try:
    import os
    import numpy as np
    import sys
    import matplotlib.pyplot as plt
    from setupNhlDataC import prepareDataC
    from scipy import signal
    import math
    from utils import small_util_routines
except ImportError:
    raise ImportError('Module not found')


critical=1
nonCritical=0

GLOBAL_VERBOSITY_LEVEL=1

utC=small_util_routines(GLOBAL_VERBOSITY_LEVEL)
VerbosityF=utC.VerbosityF
ToassertF=utC.ToassertF

#class names start uppercase, end with upper C
#function names start uppercase, end with upper F
#local class variables start with single underscore
#global class variable start with double underscore


#derived from a prepare data class, from which it derives the data directly, instead of having to give
#them as arguments


class AnalysisC():#prepareDataC):
    """main analysis class - fetches data and contains"""

    #this is a global variable - is increased for every instance for every derived class for every plot
    __globalFigureID=0
    __gesamtDatenDict={}
    __gesamtDatenGeneratedUniformUncorrelated={}
    __gesamtDatenGeneratedUnCorrelated={}
    __gesamtDatenGeneratedCorrelated={}
    __globaltestvar=0
    __EN=[]
    __ENList=[]
    __DataSetTypeList=[]
    __TeamList=[]

    def __init__(self,prepareDataC):
        self.__ENList=prepareDataC._ENList
        self.__EN=prepareDataC._EN
        self.__DataSetType=prepareDataC._DataSetType
        self.__DataSetTypeList=prepareDataC._DataSetTypeList
        self.__globaltestvar=1
        self.__gesamtDatenDict=prepareDataC._gesamtDatenDict
        self.__gesamtDatenDictSUM=prepareDataC._gesamtDatenDictSUM
        self.__gesamtDatenGeneratedUniformUncorrelated=prepareDataC._gesamtDatenGeneratedUniformUncorrelated
        self.__gesamtDatenGeneratedUnCorrelated=prepareDataC._gesamtDatenGeneratedUnCorrelated
        self.__gesamtDatenGeneratedCorrelated=prepareDataC._gesamtDatenGeneratedCorrelated

        VerbosityF(3,"gesamtDatenDict has the following keys:")
        VerbosityF(3,self.__gesamtDatenDict.keys())
        #pick out the first season data and fetch all the teams names for later reference
        self._TeamList=set([p[self.__EN.homeTeam] for p in self.__gesamtDatenDict[self.__gesamtDatenDict.keys()[0]]])

        self._currentlyAnalyzedDataType=""
        VerbosityF(0,"AnalysisC class instance successfully initialized!")

    def GetSeasonList(self,dataID):
        """return list of seasons for imported game data"""
        dataDict=self.Dataselector(dataID)
        #the season list is just a query on the keys of the dict
        return dataDict.keys()

    def GetTeamList(self):
        """get list of teams for imported game data"""
        return self._TeamList

    def GetEN(self):
        """return the enumeration object containing the data categories of imported game data"""
        return self.__EN

    def GetENList(self):
        """return the enumeration object containing the data categories of imported game data"""
        return self.__ENList

    def GetDataSetType(self):
        """return the dataset type: summed, delta or generated data"""
        return self.__DataSetType

    def GetDataSetTypeList(self):
        return self.__DataSetTypeList

    def Dataselector(self,dataID):
        """return the dataset among the bunch of delta, summed and generated datasets"""
        if dataID==self.__DataSetType.delta:
            return self.GetGesamtDaten()
        if dataID==self.__DataSetType.summed:
            return self.GetGesamtDatenSUM()
        elif dataID==self.__DataSetType.correlated:
            return self.GetGesamtDatenGenericCorrNormal()
        elif dataID==self.__DataSetType.uncorrelated_normal:
            return self.GetGesamtDatenGenericUncorrNormal()
        elif dataID==self.__DataSetType.uncorrelated_uniform:
            return self.GetGesamtDatenGenericUncorrUniform()
        else:
            ToassertF(0==1,critical,"dataselector found no valid data ID - quit")

    def GetGesamtDaten(self):
        return self.__gesamtDatenDict

    def GetGesamtDatenSUM(self):
        return self.__gesamtDatenDictSUM

    def GetGesamtDatenGenericUncorrNormal(self):
        return self.__gesamtDatenGeneratedUnCorrelated

    def GetGesamtDatenGenericUncorrUniform(self):
        return self.__gesamtDatenGeneratedUniformUncorrelated

    def GetGesamtDatenGenericCorrNormal(self):
        return self.__gesamtDatenGeneratedCorrelated

    def GetAndIncreaseGlobalFigureID(self):
        self.__globalFigureID=self.__globalFigureID+1
        return (self.__globalFigureID-1)


    def PreparePlot(self):

        plt.figure(self.GetAndIncreaseGlobalFigureID())
        plt.subplot(2,2,id)

    def PlotAll(self):

        plt.show()

    def FetchDataAndAverageOverOneGameday(self,seasonData,dataKategory):

        tmpList=[(p[dataKategory],p[self.__EN.date]) for p in seasonData]
        oldDate=-1
        zaehler=1
        id=0
        avg=0
        listOfGameDayAveragedValues=[]
        for tuple in tmpList:
            val=tuple[0]
            newDate=tuple[1]
            #average over all games in a day
            #in the first step do it differently
            if id==0:
                avg=avg+val
                zaehler=zaehler+1
            else:
                if newDate==oldDate:
                    #now we add values
                    avg=avg+val
                    zaehler=zaehler+1
                else:
                    #we do nothing but reset avg and the counter and add the previously calculated value to the list
                    avg=avg/zaehler
                    listOfGameDayAveragedValues.append(avg)
                    avg=val
                    zaehler=1
            oldDate=newDate
            id=id+1

        return listOfGameDayAveragedValues



class NeuralNetworkAnalysisC():
    """not really derived from AnalysisC, just lends some methods and shared data for all methods"""

    __LARGENEGATIVE=-np.floor(sys.float_info.max)

    def __init__(self,AnalysisC):#prepareDataC):
        #we do not instantiate the AnalysisC class, but just share some knowledge
        self.AnaC=AnalysisC
        self.__EN=self.AnaC.GetEN()
        self.__ENList=self.AnaC.GetENList()
        self._analyzedSeasonList=[]
        self._currentlyAnalyzedDataType=""
        self._currentlyAnalyzedDataDict={}
        self._seasonList=[]
        self.__teamList=self.AnaC.GetTeamList()

        self._NOHIDDENLAYERSMIN=1
        self._NOHIDDENLAYERSMAX=1
        self._NONEURONSHIDDENLAYER_MULTIPLE_OF_INPUT_NEURONS=1

        self._allHomeGamesDatesTupleT1=[]
        self._allHomeGamesDatesTupleT2=[]
        self._outData=0
        self._inputData=0
        self._homeGameT1=""
        self._homeGameT2=""

        self._outNNEstimateList=[]
        self._outNNActual=[]
        self._errNNPrct=[]

        self._inputDataSetTuple=[]
        self.__LARGENEGATIVE=-np.floor(sys.float_info.max)

        self._NN=""
        self._superVData=""
        self._trndata=""
        self._tstdata=""

#    def ExecuteNN(self):
#        [inputDatenNeuralNetwork,targetDatenNeuralNetwork]=self.SetupNNDataMatchup(gesamtDatenDict,"NYI","NYR")
#        self.ExecuteNeuralNetwork(inputDatenNeuralNetwork,targetDatenNeuralNetwork)

    def SetupNeuralNetwork(self,NhiddenLayers,NoHiddenNeuronsInNoInputLayers):

        """ setup the feed forward neural network
        :param NhiddenLayers: number of hidden layers in the neural network
        :param NoHiddenNeuronsInNoInputLayers:  number of neurons in a hidden layer
        :return: nothing, sets an internal field directing to the neural network
        """

        #the number of input layers is the number of columns of the input data
        NInputLayerNeurons=min(self._inputData.shape)
        #we experiment around with a fixed ratio of hidden to input layer neurons
        NHiddenLayerNeurons=NoHiddenNeuronsInNoInputLayers*NInputLayerNeurons
        #we only map on the outcome - win or lose
        NOutputLayerNeurons=min(self._outData.shape)

        from pybrain.structure import FeedForwardNetwork
        self._nn = FeedForwardNetwork()
        from pybrain.structure import LinearLayer, SigmoidLayer
        inLayer = LinearLayer(NInputLayerNeurons)
        hiddenLayerList=[]
        for i in range(NhiddenLayers):
            hiddenLayerList.append(SigmoidLayer(NHiddenLayerNeurons))
        outLayer = LinearLayer(NOutputLayerNeurons)
        self._nn.addInputModule(inLayer)
        #now we add all the hidden layers
        for hiddenLayerElement in hiddenLayerList:
            self._nn.addModule(hiddenLayerElement)
        self._nn.addOutputModule(outLayer)
        from pybrain.structure import FullConnection
        in_to_hidden = FullConnection(inLayer, hiddenLayerList[0])
        #connect all the hidden layers among each other - only full connections here
        List_hidden1_to_hidden2=[]
        for i in range(len(hiddenLayerList)-1):
            j=i+1
            List_hidden1_to_hidden2.append(FullConnection(hiddenLayerList[i], hiddenLayerList[j]))
        hidden_to_out = FullConnection(hiddenLayerList[-1], outLayer)
        self._nn.addConnection(in_to_hidden)
        #add all the hidden layers
        for hidden_to_hidden_conn in List_hidden1_to_hidden2:
            self._nn.addConnection(hidden_to_hidden_conn)
        self._nn.addConnection(hidden_to_out)
        self._nn.sortModules()

        VerbosityF(0,"Finished setup of a feed forward neural network with ",len(hiddenLayerList)," hidden layers and ",NHiddenLayerNeurons," neurons in the hidden layers.")

    def PrepareNNInputData(self,splitFactorTrainingTest):

        """
        split input data into training and testing set
        :param splitFactorTrainingTest: ratio of splitting into training to testing dataset
        :return nothing, writes neural network input data fields internally into array
        """

        ToassertF((0.1<=splitFactorTrainingTest) and (splitFactorTrainingTest<=0.8),critical,"in PrepareNNInputData set splitFactorTrainingTest between 0.25 and 0.75")

        from pybrain.datasets  import SupervisedDataSet

        inputSize=min(self._inputData.shape)
        outputSize=min(self._outData.shape)
        self._superVData=SupervisedDataSet(inputSize,outputSize)
        for inpElement,outElement in zip(self._inputData,self._outData):
            self._superVData.addSample(inpElement,outElement)

        #we use first 75% for training
        self._tstdata, self._trndata = self._superVData.splitWithProportion(splitFactorTrainingTest)
        #trndata._convertToOneOfMany( )
        #tstdata._convertToOneOfMany( )

        ToassertF(len(self._trndata)>0,critical,"PrepareNNInputData: length of training dataset has to be greater 0! - length of input dataset is",len(self._inputData)," splitting factor is ",splitFactorTrainingTest)

        VerbosityF(0,"Training data prepared: ")
        VerbosityF(0,"Number of training patterns: ", len(self._trndata))
        VerbosityF(0,"Number of testing datasets: ", len(self._tstdata))
        VerbosityF(0,"Input and output dimensions: ", self._trndata.indim, self._trndata.outdim)
        VerbosityF(0,"First sample:")
        VerbosityF(0,"Input data: ",self._trndata['input'][0])
        VerbosityF(0,"Output data: ", self._trndata['target'][0])

    def RunNeuralNetwork(self):

        """
        setup the neural network
        input:
            inputData:
                    pre-formatted data, with d-columns - we initialize as many input neurons as data columns
            NhiddenLayers:
                    number of hidden (in between) layers of the neural network
            NoHiddenNeuronsInNoInputLayers:
                    number of neurons in hidden layer as multiple of input neurons
        output:
            estimationOut
                    list of estimated outputs
            actualOut
                    list of actual outputs as given in input
            errorPercent
                    error of estimation in %
        """

        from pybrain.supervised.trainers import BackpropTrainer

        trainer = BackpropTrainer( self._nn, dataset=self._trndata, momentum=0.1, verbose=False, weightdecay=0.01)
        #trainer.setData(superVData)

        from pybrain.tools.validation import ModuleValidator
        #from pybrain.tools.validation   import NNregression

        #contains methods to calculate error between predicted and target values
        MV=ModuleValidator()

        #NNreg=NNregression(superVData)
        #for i in range(2):
        #    trainer.trainEpochs( 300 )
            #now we check the results visually
            #for (elemI,elemO) in zip(inputData[-5:],outData[-5:]):
            #    print nn.activate(elemI),elemO
            #sqErr=MV.ESS(out,targ)
            #print "number of direct hits: ",MV.classificationPerformance(out,targ)


            #print "epoch: %4d" % trainer.totalepochs, \
            #      "  train error: %5.2f%%" % trnresult, \
            #      "  test error: %5.2f%%" % tstresult

        trainer.trainUntilConvergence(dataset=None, maxEpochs=500, verbose=False, continueEpochs=100)#, validationProportion=0.25)
        nnoutList=[];actualOutList=[]
        # for (elemI,elemO) in zip(self._inputData,self._outData):
        #     nnoutList.append(self._nn.activate(elemI))
        #     actualOutList.append(elemO)

        for (elemI,elemO) in zip(self._tstdata["input"],self._tstdata["target"]):
            nnoutList.append(self._nn.activate(elemI))
            actualOutList.append(elemO)

        #we prepare the neural-network output from real space to 0 and 1:
        estimationOut=[]
        for liele in nnoutList:
            estimationOut.append(math.ceil(liele-np.mean(nnoutList)))

        tmp=0
        for (eleNO,eleAO) in zip(estimationOut,actualOutList):
            tmp=tmp+abs(eleNO-eleAO)/len(actualOutList)
            errorPercent=100.*(1.-tmp)

        VerbosityF(0,"The neuronal network predicted ",round(errorPercent[0]), " % of the test data correctly")

        return estimationOut,actualOutList,errorPercent

    def SetHiddenNeuronsAsMultipleOfInputNeurons(self,inputNumber):
        ToassertF(type(inputNumber)==int,critical,"input to SetHiddenNeuronsAsMultipleOfInputNeurons has to be an int")
        self._NONEURONSHIDDENLAYER_MULTIPLE_OF_INPUT_NEURONS=inputNumber

    def GetHiddenNeuronsAsMultipleOfInputNeurons(self):
        return self._NONEURONSHIDDENLAYER_MULTIPLE_OF_INPUT_NEURONS

    def SetNoHiddenNeuronLayersMinMax(self,inputMin,inputMax):
        ToassertF(type(inputMax)==int and type(inputMin)==int,critical,"input to SetHiddenNeuronsAsMultipleOfInputNeurons has to be an int")
        self._NOHIDDENLAYERSMIN=inputMin
        self._NOHIDDENLAYERSMAX=inputMax

    def ExecuteNeuralNetworkConfigurations(self,inputDatenNeuralNetwork,targetDatenNeuralNetwork):

        #first set up the whole dataset for team1 and team2
        #now we estimate using 4 different neural network sizes, with different number of hidden layers
        outEstimate=[]
        outActual=[]
        errPrct=[]
        for noNeuronsInHiddenLayer in range(1,self._NONEURONSHIDDENLAYER_MULTIPLE_OF_INPUT_NEURONS+1):
            for nohiddenLayers in range(self._NOHIDDENLAYERSMIN,self._NOHIDDENLAYERSMAX+1):
            #with 1 hidden Layer
                [tmp1,tmp2,tmp3]=self.RunNeuralNetwork()
                self._outNNEstimateList.append(tmp1)
                self._outNNActual.append(tmp2)
                self._errNNPrct.append(tmp3)

    def PlotNeuralNetworkConfigurationResults(self):

        for noNeuronsInHiddenLayer in range(1,self._NONEURONSHIDDENLAYER_MULTIPLE_OF_INPUT_NEURONS+1):
            for nohiddenLayers in range(self._NOHIDDENLAYERSMIN,self._NOHIDDENLAYERSMAX):

                print "results of a neural network conditioning with ",nohiddenLayers," hidden layers and "
                print "number of neurons in hidden layers: ",noNeuronsInHiddenLayer," times the neurons in the input layer"
                print "in the prediction of the full dataset ",self._errNNPrct[nohiddenLayers], "% of the data has been correctly predicted"

                plotid=(noNeuronsInHiddenLayer-1)*(self._NOHIDDENLAYERSMAX-self._NOHIDDENLAYERSMIN)+nohiddenLayers
                NOPLOTS=self._NONEURONSHIDDENLAYER_MULTIPLE_OF_INPUT_NEURONS*self._NOHIDDENLAYERSMAX
                plt.subplot(NOPLOTS/2.,NOPLOTS/2.+np.mod(NOPLOTS,2),plotid)
                ax=plt.plot(self._outNNEstimateList[nohiddenLayers],'o')
                plt.plot(self._outNNActual[nohiddenLayers])
                axes = plt.gca()
                axes.set_ylim([-0.5,1.5])

        plt.show()

    def SetCurrentlyAnalyzedDataSetDict(self,typeString):
        """as specified in input, read the input dataset from analysis class"""
        """input:
                string: type
                can be "delta","summed","correlated","uncorrelated_normal","uncorrelated_uniform
           no output
        """

        self._currentlyAnalyzedDataDict=self.AnaC.Dataselector(typeString)

    def AnalyzeMatchup(self,Team1Name,Team2Name,gesamtDatenTyp=1):

        #self._currentlyAnalyzedDataType=self.AnaC.GetDataSetTypeList()[gesamtDatenTyp]
        #gesamtDatenDict=self.AnaC.Dataselector(gesamtDatenTyp)
        #choose the delta value dataset
        self.SetCurrentlyAnalyzedDataSetDict(0)
        self.SetupNNDataMatchup(Team1Name,Team2Name)
        self.ExecuteNeuralNetworkConfigurations(self,self._inputData,self._targetData)
        self.PlotNeuralNetworkConfigurationResults()

    def SetupNNDataMatchup(self,Team1,Team2):
        """input:
                :param gesamtDatenDict
                            whole dataset for all seasons and stat categories
                :param Team1Name
                            name of team1 of matchup
                :param Team2Name:
                            name of team1 of matchup
            output: returns interna fields for
                _inputData
                            multivariate data with rows containing data for each matchup
                            and columns containing categorical data as follows:
                                #1 first team home matchup score, until current matchup - add only information of one match at time
                                #2 second team home matchup score (same as first team away matchup score), until current matchup
                                #3 first teams performance over last N games
                                #4 second teams performance over last N games
                                #5 days off the first team had before matchup
                                #6 days off the second team had before matchup
                _outData
                            the target data, in this case a one dimensional array
                            containing integer 1 if team1 won, else 0

        """

        ToassertF(not self._currentlyAnalyzedDataDict==[],critical,"no dictionary for evaluaten chosen - call method SetCurrentlyAnalyzedDict before!")

        #we concatenate all the data at our
        allseasonData=[]

        self._allHomeGamesDatesTupleT1=[]
        self._allHomeGamesDatesTupleT2=[]
        self._inputDataSetTuple=[]

        seasonList=sorted(self._currentlyAnalyzedDataDict.keys())

        for seasonKey in seasonList:
            allseasonData=allseasonData+self._currentlyAnalyzedDataDict[seasonKey]
            """the neural network input dataset has an entry for every matchup of the two teams"""
            """get the unique identifier for every matchup - the triple, the id, the date and the season we are in"""
            homeGameT1DateTupleList =[(p[self.__EN.ID],p[self.__EN.date],p[self.__EN.season]) for p in self._currentlyAnalyzedDataDict[seasonKey] if Team1==p[self.__EN.homeTeam] and Team2==p[self.__EN.awayTeam]]
            homeGameT2DateTupleList =[(p[self.__EN.ID],p[self.__EN.date],p[self.__EN.season]) for p in self._currentlyAnalyzedDataDict[seasonKey] if Team2==p[self.__EN.homeTeam] and Team1==p[self.__EN.awayTeam]]
            #numberOfMatchupsThisSeason=len(homeGameT1Date)+len(homeGameT2Date)
            thisSeasonsTuple=homeGameT1DateTupleList+homeGameT2DateTupleList
            #now sort the matchups after id, the first entry
            thisSeasonsTupleSorted=sorted(thisSeasonsTuple,key=lambda row: row[0])
            #now we have an ordered
            self._inputDataSetTuple=self._inputDataSetTuple+thisSeasonsTupleSorted
            #now the direct mapping from home games to dates and ids in two separate list of tuples
            self._allHomeGamesDatesTupleT1=self._allHomeGamesDatesTupleT1+homeGameT1DateTupleList
            self._allHomeGamesDatesTupleT2=self._allHomeGamesDatesTupleT2+homeGameT2DateTupleList


        #generate the networks output data
        #it consists in a lone number, 1 for win and 0 for loss of Team1
        self._outData=np.zeros([len(self._inputDataSetTuple),1])
        for index in range(len(self._inputDataSetTuple)):
            #we read all the entries of outData in the same order as that of the inputData!
            id=self._inputDataSetTuple[index][0]
            date=self._inputDataSetTuple[index][1]
            season=self._inputDataSetTuple[index][2]
            #sets the wins for Team1 - so 0 for a loss, 1 for a win for team 1
            for element in self._currentlyAnalyzedDataDict[season]:
                if (id==element[self.__EN.ID] and season==element[self.__EN.season]):
                    self._outData[index]=element[self.__EN.wonYN]

        #"""now merge the datasets"""
        ##totalMatchups=homeGameT1Date+homeGameT2Date
        #"""then sort them"""
        #dateMatchups=sorted(totalMatchups)

        """inputData: array with all the matchups
                        has dimension of (all matchups x extracted features)
                        initialized to the largest negative number"""
        #in rows the various games are stored, as numpy calls the array rowwise
        #have six columns for input data
        #we fill it with large negatives to identify non-filled areas
        self._inputData=np.full([len(self._inputDataSetTuple),6],self.__LARGENEGATIVE)

        self._homeGameT1=[p[self.__EN.wonYN] for p in allseasonData if Team1==p[self.__EN.homeTeam] and Team2==p[self.__EN.awayTeam]]
        self._homeGameT2=[p[self.__EN.wonYN] for p in allseasonData if Team2==p[self.__EN.homeTeam] and Team1==p[self.__EN.awayTeam]]

        """lists contain the chance for a win, assuming uniform distribution, according to games played up to the current date
        corresponds to #1 and #2
        """
        winHomeChanceT1=self.GetWinChanceHelper(self._homeGameT1)
        winHomeChanceT2=self.GetWinChanceHelper(self._homeGameT2)

        """now read the result in the global input data structure"""
        """we use the mapping arrays for team1 and team2 home games"""

        for index in range(len(self._inputDataSetTuple)):
            id=self._inputDataSetTuple[index][0]
            date=self._inputDataSetTuple[index][1]
            season=self._inputDataSetTuple[index][2]
            #it might not be found in homegames - then do nothing
            try:
                #no exception handling, as we are just going on
                gotHomeIndex1=self._allHomeGamesDatesTupleT1.index((id,date,season))
                self._inputData[index,0]=winHomeChanceT1[gotHomeIndex1]
            except:
                1==1
            try:

                gotHomeIndex2=self._allHomeGamesDatesTupleT2.index((id,date,season))
                self._inputData[index,1]=winHomeChanceT2[gotHomeIndex2]
            except:
                1==1

            #the first two rows are now filled with data
            #the entries in between are now minus large constant - we have to fill them up with the latest value to remain consistent
            #now do the mapping from home games to overall games

        """the first and seconds team performance over the last N games"""

        """
        now lets call the function to evaluate for a certain id and season the teams performance over the last couple of games
        as well as reading the difference in the ids to get the days between the last match of the team and the current matchup
        """
        N=5
        #this is slow (loop) and non-pythonic but readable
        for index in range(len(self._inputDataSetTuple)):
            id=self._inputDataSetTuple[index][0]
            date=self._inputDataSetTuple[index][1]
            season=self._inputDataSetTuple[index][2]
            [performanceTeam1,dateOfLastGame1]=self.AssessLastNGamesHelper(date,season,Team1,allseasonData,N)
            [performanceTeam2,dateOfLastGame2]=self.AssessLastNGamesHelper(date,season,Team2,allseasonData,N)
            self._inputData[index,2]=performanceTeam1
            self._inputData[index,3]=performanceTeam2
            self._inputData[index,4]=(date-dateOfLastGame1)
            self._inputData[index,5]=(date-dateOfLastGame2)


        """fillUpZeroElementsInInputData: we fill up the data which has not been written in the input dataset for the neural network
        unfilled values, which are filled"""

        self.FillUpZeroElementsInInputDataHelper(self._inputData)

        VerbosityF(0,"Finished the preparation of input data for Neural Network.")
        VerbosityF(0,"The input dataset contains ",str(len(self._inputDataSetTuple))," samples ")
        VerbosityF(0,"We have ",len(self._inputData[:,1])," input datasets and one output dataset")

        #return inputData,outData


    def FillUpZeroElementsInInputDataHelper(self,inputData):
        noColumns=inputData.shape[1]
        lastfiniteValue=np.zeros(noColumns)
        for (i,row) in zip(range(len(inputData)),inputData):
            for (j,column) in zip(range(len(row)),row):
                #if the entry is a large negative number, we have to fill up the value with that of the ls
                if column==self.__LARGENEGATIVE:
                    inputData[i][j]=lastfiniteValue[j]
                else:
                    lastfiniteValue[j]=column


    def AssessLastNGamesHelper(self,currentDate,currentSeason,Team,fullData,NLastGames):
        """assess the performance of the team for the last N games
        starting from
        input   id
        input   season for team
        input   Team
        input   full dataset over all seasons
        input   NLastGames: number of games to sum over (shifted sum)
        output  number of team wins in last N performances
        """
        #read tuple with wins and ID -> from id extract the id of last game played
        gameResultsBeforeDate=[(p[self.__EN.wonYN],p[self.__EN.date]) for p in fullData if ((Team==p[self.__EN.homeTeam]) and \
            (currentSeason==p[self.__EN.season] and p[self.__EN.date]<currentDate)) ]
        #wins for the away team have to be inverted 0--->1; add one!
        tmpTupleList=[(p[self.__EN.wonYN],p[self.__EN.date]) for p in fullData if ((Team==p[self.__EN.awayTeam]) and \
            (currentSeason==p[self.__EN.season] and p[self.__EN.date]<currentDate)) ]
        #now invertResults an away loss is 0 -> convert to one
        invertedGameResultsBeforeDate=[]
        for tuple in tmpTupleList:
            if tuple[0]==0:
                invertedGameResultsBeforeDate.append((1,tuple[1]))
            else:
                invertedGameResultsBeforeDate.append((0,tuple[1]))

        gameResultsBeforeDate=gameResultsBeforeDate+invertedGameResultsBeforeDate
        #at index [-1][1] we find the last tuple in the list, and there the id!
        idOfLastGamePlayed=0
        for (tuple1,tuple2) in zip(gameResultsBeforeDate,invertedGameResultsBeforeDate):
            id1=tuple1[1];id2=tuple2[1];
            idOfLastGamePlayed=max(idOfLastGamePlayed,id1)
            idOfLastGamePlayed=max(idOfLastGamePlayed,id2)

        """
        output:
        the number of wins in the last N games for the data filtered!
        This is a slow but readable approach
        summing weighs every win with one, every loss with zero
        """
        #the zero index in the sum aims at the EN.wonYN statistics, the first entry in the tuple
        sumt=0
        for tuple in gameResultsBeforeDate[-NLastGames:]:
            sumt=sumt+tuple[0]

        return sumt,idOfLastGamePlayed


    def GetWinChanceHelper(self,homeGame):
        """
        input:
                list of 0 and 1, 1 for win, 0 for loss
        output:
                list with sums up to the list index
        """
        returnList=[]
        addedScore=0
        for (i,ele) in zip(range(1,len(homeGame)+1),homeGame):
            addedScore=addedScore+ele
            #calculate probability assuming a uniform distribution
            #divide by all elements until now, assuming uniform distribution
            returnList.append(float(addedScore)/float(i))
        return returnList

class WaveletAnalysisC():
    """not really derived from AnalysisC, just lends some methods and shared data for all methods"""

    def __init__(self,AnalysisC):#prepareDataC):
        #we do not instantiate the AnalysisC class, but just share some knowledge
        self.AnaC=AnalysisC
        self.__EN=self.AnaC.GetEN()
        self._waveletInputDataListofList=[]
        self._cwtmatrListOfLists=[]
        self.__ENList=self.AnaC.GetENList()
        self._analyzedSeasonList=[]
        self._currentlyAnalyzedDataType=""


    def PrepareTheData(self,gesamtDatenDict,dataKategory):

        for seasonKey in gesamtDatenDict.keys():
            seasonData=gesamtDatenDict[seasonKey]
            self._analyzedSeasonList.append(seasonKey)
            self._waveletInputDataListofList.append(self.AnaC.FetchDataAndAverageOverOneGameday(seasonData,dataKategory))

    def Analyze(self,kategory,gesamtDatenTyp=1):

        self._currentlyAnalyzedDataType=self.AnaC.GetDataSetTypeList()[gesamtDatenTyp]
        gesamtDatenDict=self.AnaC.Dataselector(gesamtDatenTyp)
        self.PrepareTheData(gesamtDatenDict,kategory)
        self.DoTheWaveletTransformForList()
        self.PlotWaveletList(kategory)
        #self.PrepareTheData(self.AnaC.GetDataSetType().original,self.__EN.PenaltyminutesDiff)
        #self.DoTheWaveletTransformForList()
        #self.PlotWaveletList()

    def AnalyzePIM(self):
        self.PrepareTheData(self.AnaC.GetDataSetType().original,self.__EN.PenaltyminutesDiff)
        self.DoTheWaveletTransformForList()
        self.PlotWaveletList()

    def DoTheWaveletTransformForList(self):
        for dataForOneSeason in self._waveletInputDataListofList:
            self._cwtmatrListOfLists.append(self.DoTheWaveletTransform(dataForOneSeason))

    def DoTheWaveletTransform(self,inputData):
        widths = np.arange(1, 5)
        tmpCwtmatr = signal.cwt(inputData, signal.ricker, widths)
        return tmpCwtmatr

    def PlotWaveletList(self,kategory):
        for (season,transformedDataSet) in zip(self._analyzedSeasonList,self._cwtmatrListOfLists):
            title="".join([str(self.__ENList[kategory])," | ",self._currentlyAnalyzedDataType," | season ",str(season)])
            self.PlotWavelets(transformedDataSet,title)

    def PlotWavelets(self,inputData,titleMessage):

        fig=plt.figure(self.AnaC.GetAndIncreaseGlobalFigureID())
        inputTransformed=np.asarray(inputData)
        #cmap='PRGn'
        plt.imshow(inputTransformed, cmap='hot', aspect='auto',vmax=abs(inputTransformed).max(), vmin=-abs(inputTransformed).max())
        plt.title(titleMessage)
        plt.xlabel("game day [d]")
        plt.ylabel("wavelet component")
        plt.colorbar()

class BivariateAnalysisC():
    """not really derived from AnalysisC, just lends some methods and shared data for all methods"""

    def __init__(self,AnalysisC):#prepareDataC):
        #we do not instantiate the AnalysisC class, but just share some knowledge
        self.AnaC=AnalysisC
        #AnalysisC.__init__(self,prepareDataC)
        self._listDtLost=[]
        self._listDtWon=[]
        self._corrAboveThresholdDict={}

        #self._gesamtDatenDict=AnalysisCI.__gesamtDatenDict

    def ComputeAll(self):
        #self.computeHistogramsHelper()
        self.ComputeTimeWinStatistics()

    def ComputeTimeWinStatistics(self,dataTypeID):
        """compute the histograms of data in a season for two datasets - bivariate"""
        """
        input: dataType:
        """

        #ToassertF(AnalysisC.__gesamtDatenDict=={},critical,"general data dictionary _gesamtDatenDict is empty")

        EN=self.AnaC.GetEN()
        gesamtDatenDict=self.AnaC.Dataselector(dataTypeID)
        #VerbosityF(2,"dataType ",dataTypeID," gesDaten erster Datensatz: ",gesamtDatenDict[2010][0:3])
        self._listDtWon=[];self._listDtLost=[]
        for (i,seasonKey) in zip(range(1,len(gesamtDatenDict.keys())+1),sorted(gesamtDatenDict.keys())):
            seasonData=gesamtDatenDict[seasonKey]
            Teams=set([p[EN.homeTeam] for p in seasonData])
            #have to split up for teams, as statistic is evaluated separately for every team
            for team in Teams:
                listWonYNDate=[(p[EN.date],p[EN.wonYN]) for p in seasonData if team==p[EN.homeTeam] or team==p[EN.awayTeam]]
                it=0;listDtWon=[]#[0]*max(len(seasonData),len(seasonData[0]));
                #[0]*max(len(seasonData),len(seasonData[0]));listWon=[];
                for element in listWonYNDate:
                    date=element[0]
                    if it==0:
                        prevDate=date
                    deltaDate=date-prevDate
                    if deltaDate<10 and deltaDate>0:
                        if element[1]==1:
                        #team has won
                            self._listDtWon.append(deltaDate)
                        else:
                            self._listDtLost.append(deltaDate)
                    prevDate=date
                    it=it+1

    def ReturnWinList(self):
        return self._listDtWon

    def ReturnLossList(self):
        return self._listDtLost


    def CalcAndPlot2HistogramsBivariate(self,histoData1,histoData2,message,labelX,labelY):

        plt.figure(self.AnaC.GetAndIncreaseGlobalFigureID())
        wgh1 = np.ones_like(histoData1)/float(len(histoData1))
        wgh2 = np.ones_like(histoData2)/float(len(histoData2))
        plt.title("".join([message]))
        plt.hist(histoData1, bins=20,weights=wgh1, histtype='stepfilled', normed=False, color='b', label='win')
        plt.xlabel(labelX)
        plt.hist(histoData2, bins=20,weights=wgh2, histtype='stepfilled', normed=False, color='r', alpha=0.5, label='loss')
        plt.ylabel(labelY)
        #plt.xlabel("days between games [d]")
        #plt.ylabel("probability of event")
        plt.legend()


    def PlotScatterBivariate(self,dataTypeID,EnumData1,EnumData2):

        gesamtDatenDict=self.AnaC.Dataselector(dataTypeID)
        for (i,seasonKey) in zip(range(1,len(gesamtDatenDict.keys())+1),sorted(gesamtDatenDict.keys())):
            #print i,seasonKey
            plt.figure(self.AnaC.GetAndIncreaseGlobalFigureID())
            seasonData=gesamtDatenDict[seasonKey]
            hits=[p[EnumData1] for p in seasonData]
            pim=[p[EnumData2] for p in seasonData]
            #ax = plt.subplot(len(gesamtDatenDict.keys())/2,2,i)
            #plt.subplots_adjust(wspace=0.4,hspace=0.4)
            #ax.set_xlabel('penalty minutes')
            #ax.set_ylabel('hits')
            #plt.title("".join("season "+str(seasonKey)))
            plt.scatter(hits,pim)
        #    plt.legend(["pim in home wins","pim in home losses"])


    def ScanForBivariateCorrelations(self,dataTypeID,crossCorrelationThreshold=0.3):

        gesamtDatenDict=self.AnaC.Dataselector(dataTypeID)
        collectedCovList=[]
        EN=self.AnaC.GetEN()
        for (i,seasonKey) in zip(range(1,len(gesamtDatenDict.keys())+1),sorted(gesamtDatenDict.keys())):
            seasonData=gesamtDatenDict[seasonKey]
            Teams=set([p[EN.homeTeam] for p in seasonData])
            captionList=["WonYN","ScoreDiff","ShDiff","FaceoffDiff","TakeawayDiff", \
             "GiveawaysDiff","PenaltyminutesDiff","HitsDiff","powerplayGoalsDiff"]
            foundValues=0
            for team in Teams:
                data=[np.asarray([p[\
                        EN.wonYN],\
                        p[EN.ScoreDiff],\
                        p[EN.ShDiff],\
                        p[EN.FaceoffDiff],\
                        p[EN.TakeawayDiff],\
                        p[EN.GiveawaysDiff],\
                        p[EN.PenaltyminutesDiff],\
                        p[EN.HitsDiff],\
                        p[EN.powerplayGoalsDiff],\
                       ]) for p in seasonData if team==p[EN.homeTeam] or team==p[EN.awayTeam]]
                dataArray=np.asarray(data)
                collectedCov=[]
                #start at index 1 - omit correlation of score and win
                for i in range(1,min(dataArray.shape)):
                    for j in range(i+1,min(dataArray.shape)):
                            data1=dataArray.T[i,:]
                            data2=dataArray.T[j,:]
                            #print "mean:","\t",'{:.2f}'.format(np.mean(hitsD[team])),"|",'{:.2f}'.format(np.mean(pimD[team]))
                            #print "standard deviation: ","\t",'{:.2f}'.format(np.std(hitsD[team])),"|",'{:.2f}'.format(np.std(pimD[team]))
                            #print "skewness: ","\t",'{:.2f}'.format(stats.skew(hitsD[team])),"|",'{:.2f}'.format(stats.skew(pimD[team]))
                            #print "kurtosis: ","\t",'{:.2f}'.format(stats.kurtosis(hitsD[team])),"|",'{:.2f}'.format(stats.kurtosis(pimD[team]))
                            ddata=np.asarray([data1,data2])
                            calcCov=np.corrcoef(ddata)[0][1]
                            if abs(calcCov)>crossCorrelationThreshold:
                                foundValues=foundValues+1
                                collectedCov.append(calcCov)
                                kategory1=captionList[i]
                                kategory2=captionList[j]
                                teamTmpDict={}
                                teamTmpDict["type"]="scatter"
                                teamTmpDict["correlCoeff"]=calcCov
                                #teamTmpDict["team"]=team
                                teamTmpDict["dataX"]=data1
                                teamTmpDict["dataY"]=data2
                                teamTmpDict["titleString"]="".join([str(seasonKey),"\t",str(team),"\t",kategory1," vs.",kategory2])
                                #if not (kategory1,kategory2) in self._corrAboveThresholdZaehlerDict:
                                #    self._corrAboveThresholdZaehlerDict[(kategory1,kategory2)]=0
                                #else:
                                #    self._corrAboveThresholdZaehlerDict[(kategory1,kategory2)]=self._corrAboveThresholdZaehlerDict[(kategory1,kategory2)]+1
                                #id=self._corrAboveThresholdZaehlerDict[(kategory1,kategory2)]
                                self._corrAboveThresholdDict[(kategory1,kategory2,team)]=teamTmpDict
                                #plt.figure(self.AnaC.GetAndIncreaseGlobalFigureID())
                                #plt.scatter(data1,data2)
                                #plt.title("".join([str(seasonKey),"\t",str(team),"\t",captionList[i],"\t",captionList[j]]))
                                VerbosityF(3,"for team: ",team," the crosscorrelation threshold of ", crossCorrelationThreshold,\
                                " is surpassed for categories ", captionList[i]," and ", captionList[j],"!")
                                VerbosityF(3,"crosscorrelation value is ",calcCov)
                        #for every team append the covariance value
                #collectedCovList.append(collectedCov)
        VerbosityF(3,"We found ",foundValues, " bivariate datapairs with higher correlation then threshold of ",crossCorrelationThreshold)


    def PlotAllFromDict2D(self,dictionary,titleSupplement):

        """import a 2d dictionary and plot all data it contains"""

        harvesterOfTuples=[]

        for keyTuple in dictionary.keys():
            #we assemble a list of tuples already visited
            if keyTuple not in harvesterOfTuples:
                subDict=dictionary[keyTuple]
                xAxisString=keyTuple[0]
                yAxisString=keyTuple[1]
                team=keyTuple[2]
                harvesterOfTuples.append(keyTuple)
                if isinstance(subDict,dict):
                    if "type" in subDict:
                        plotType=subDict["type"]
                    else:
                        ToassertF(1==0,critical,"did not find key type in plotting dictionary - quit")

                    titleString=subDict["titleString"]
                    #if "team" in subDict:
                        #team=subDict["team"]
                        #legendStringList.append(team)

                    if plotType=="scatter":

                            fig=plt.figure(self.AnaC.GetAndIncreaseGlobalFigureID())
                            ax=fig.add_subplot(111)
                            plt.title("".join([titleSupplement,titleString]))
                            plt.xlabel(xAxisString)
                            plt.xlabel(yAxisString)

                            #now get all the teams value for this pair of data lists

                            for team in self.AnaC._TeamList:

                                if (xAxisString,yAxisString,team) in dictionary:
                                    z=float(np.random.uniform(0, 1, size=1))
                                    green=[0.0, 0.5, 0.0]
                                    red=(1.0, 0.0, 0.0)
                                    thirdC=(0.0, 0.0, 0.5)
                                    #zcolor=z*green#+(1-z)*red+(0.5-z)*thirdC
                                    zcolor=(float(np.random.uniform(0, 1, size=1)),float(np.random.uniform(0, 1, size=1)),float(np.random.uniform(0, 1, size=1)))
                                    fetchForTuple=(xAxisString,yAxisString,team)
                                    subDi=dictionary[fetchForTuple]
                                    #now add the tuple to the exclusion list
                                    harvesterOfTuples.append(fetchForTuple)
                                    dataX=subDi["dataX"]
                                    dataY=subDi["dataY"]
                                    ax.scatter(dataX,dataY,c=zcolor,s=50,label=team)

                            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);

class MultivariateStatisticsC():

    """not really derived from AnalysisC, just lends some methods and shared data for all methods"""

    def __init__(self,AnalysisC):#prepareDataC):

        try:
            from sklearn.svm import SVC
            from sklearn.metrics.pairwise import chi2_kernel,laplacian_kernel,linear_kernel,polynomial_kernel,rbf_kernel

        except ImportError:
            pass

        #we do not instantiate the AnalysisC class, but just share some knowledge
        self.AnaC=AnalysisC
        #AnalysisC.__init__(self,prepareDataC)
        self._listDtLost=[]
        self._listDtWon=[]
        self._corrAboveThresholdDict={}
        self._currentlyAnalyzedDataDict={}
        self._mappingMatrix=0
        self.__EN=self.AnaC.GetEN()
        self.__ENList=self.AnaC.GetENList()
        self.__teamList=self.AnaC.GetTeamList()
        self._compactMultivariateData={}
        self._compactMultivariateDataAllSeasons={}
        self.linear_kernel=linear_kernel
        self.rbf_kernel=rbf_kernel
        self.minPath=[]

    def GetData(self,gesamtDatenTyp=1):

        self._currentlyAnalyzedDataType=self.AnaC.GetDataSetTypeList()[gesamtDatenTyp]
        self._currentlyAnalyzedDataDict=self.AnaC.Dataselector(gesamtDatenTyp)


    def GetMatchupDataBetweenTeamsAllSeasons(self,team1,team2):
        """compile the data of matchups between two teams over multiple seasons
        and store in self._currentlyAnalyzedDataDict"""

        allseasonData=[]
        for seasonKey in self.AnaC.GetSeasonList():
            allseasonData=allseasonData+self._currentlyAnalyzedDataDict[seasonKey]
            """the neural network input dataset has an entry for every matchup of the two teams"""
            """get the unique identifier for every matchup - the triple, the id, the date and the season we are in"""
        GameDataList =[p for p in allseasonData if ((team1==p[self.__EN.homeTeam] and team2==p[self.__EN.awayTeam]) or (team2==p[self.__EN.homeTeam] and team1==p[self.__EN.awayTeam])) ]
        self._currentlyAnalyzedDataDict={}
        self._currentlyAnalyzedDataDict["all seasons - matchup"]=GameDataList
            #homeGameT2DataList =[(p for p in self._currentlyAnalyzedDataDict[seasonKey] if team2==p[self.__EN.homeTeam] and team1==p[self.__EN.awayTeam]]
            #numberOfMatchupsThisSeason=len(homeGameT1Date)+len(homeGameT2Date)
            #thisSeasonsTuple=homeGameT1DateTupleList+homeGameT2DateTupleList
            #now sort the matchups after id, the first entry
            #thisSeasonsTupleSorted=sorted(thisSeasonsTuple,key=lambda row: row[0])

    def PrepareData(self):

        """we read the 9 categories for use in multivariate analysis"""

        for seasonKey in self._currentlyAnalyzedDataDict.keys():
            seasonData=self._currentlyAnalyzedDataDict[seasonKey]
            for (index,team) in zip(range(len(self.__teamList)),self.__teamList):
                self._compactMultivariateData[team,seasonKey]=[np.asarray([p[\
                        self.__EN.date],\
                        #p[self.__EN.wonYN],\
                        p[self.__EN.ScoreDiff],\
                        p[self.__EN.ShDiff],\
                        p[self.__EN.FaceoffDiff],\
                        p[self.__EN.TakeawayDiff],\
                        p[self.__EN.GiveawaysDiff],\
                        p[self.__EN.PenaltyminutesDiff],\
                        p[self.__EN.HitsDiff],\
                        p[self.__EN.powerplayGoalsDiff],\
                       ]) for p in seasonData if team==p[self.__EN.homeTeam] or team==p[self.__EN.awayTeam]]

                if team not in self._compactMultivariateDataAllSeasons:
                    self._compactMultivariateDataAllSeasons[team]=self._compactMultivariateData[team,seasonKey]
                else:
                    self._compactMultivariateDataAllSeasons[team]=self._compactMultivariateDataAllSeasons[team]+self._compactMultivariateData[team,seasonKey]

    def KernelMapping(self,kernelName,team,gesamtDatenTyp):
        """do a kernel mapping from KxN space to NxN space
        by applying various mapping functions to the input data"""

        self.GetData(gesamtDatenTyp)
        self.PrepareData()
        self.MultivariateKernelMappingOneTeam(kernelName,team)
        #self.PlotDistanceMatrix2D()
        self.FindMinPathThrougData()
        self.PlotDistanceMatrix2D()
        self.PlotDistanceMatrix3D()


    def AnalyzeMatchup(self,kernelName,team1,team2,gesamtDatenTyp):

        self.GetData(gesamtDatenTyp)
        self.GetMatchupDataBetweenTeamsAllSeasons()
        self.PrepareData()
        self.MultivariateKernelMappingOneTeam(kernelName,team)

    def MultivariateKernelMappingOneTeam(self,kernelName,team):

        self.KernelWrapper(kernelName,team)

    def PlotDistanceMatrix2D(self):
        """plot the mapped data against the date"""
        plt.plot(self.minPath,"o")
        plt.plot(self.minPath)

    def PlotDistanceMatrix3D(self):
        """plot the mapped data against the date"""
        plt.matshow(self._mappingMatrix)

    def FindMinPathThrougData(self):

        exclusionList=[]
        outputIndexList=[]
        outputIndexTupleList=[]
        for (rowindex,row) in zip(range(len(self._mappingMatrix)),self._mappingMatrix):
            VerbosityF(0,"len row",len(row),"len row without duplicates: ",len(set(row)))
            #VerbosityF(0,"row with index ",rowindex," ",row)
            #first exclude the diagonal element
            #exclusionList.append(row[rowindex])
            #list of all values not in exclusion list
            rowLi=list(row)
            #plt.plot(row)
            rowFilteredLi=[p for p in rowLi if p not in exclusionList]
            #print "row index ",rowindex," len row ", len(rowLi), " len filter list ", len(rowFilteredLi), " len exclusion list ",len(exclusionList)
            #look for the index of the maximal element of the filtered list in the total list
            index=rowLi.index(max(rowFilteredLi))
            #add this element to the exclusion list
            exclusionList.append(row[index])
            #and to the output list
            outputIndexTupleList.append((rowindex,index))
            outputIndexList.append(index) #(index)
            #for i in range(len(row)):
            #if index not in exclusionList:
            #exclusionList.append(index)

        self.minPath=outputIndexList
        #plt.show()
        VerbosityF(0,"len minPath",len(self.minPath),"len without duplicates: ",len(set(self.minPath)))
        self.PhasePlot(self.minPath)


    def PhasePlot(self,inputList):
        deltaInputList=[0]
        for i in range(1,len(inputList)):
            deltaInputList.append(inputList[i]-inputList[i-1])

        fig=plt.figure(self.AnaC.GetAndIncreaseGlobalFigureID())
        plt.plot(inputList,deltaInputList)

    def KernelWrapper(self,kernelName,team):

        dataToAnalyze=[];dateList=[];winList=[]
        for listElement in self._compactMultivariateDataAllSeasons[team]:
            winList.append(listElement[1])
            dateList.append(listElement[0])
            dataToAnalyze.append(listElement[2:])

        if kernelName=="linear":
            tmpMatrix=self.LinearKernelMapping(team,dataToAnalyze)
        if kernelName=="rbf":
            tmpMatrix=self.RBFKernelMapping(team,dataToAnalyze)

        self._mappingMatrix=tmpMatrix#-np.diag(tmpMatrix)
        self._mappingMatrix=abs(self._mappingMatrix)

    def LinearKernelMapping(self,team,dataToAnalyze):
        """the linear kernel mapping multiplies each dataset(row, with N features) with the transposed of the next set
        is zero if datasets are orthogonal (in the space spanned by the unit vectors of input space)"""

        tmpMatrix=self.linear_kernel(dataToAnalyze)
        return tmpMatrix

    def RBFKernelMapping(self,team,dataToAnalyze):
        """the linear kernel mapping multiplies each dataset(row, with N features) with the transposed of the next set
        is zero if datasets are orthogonal (in the space spanned by the unit vectors of input space)"""

        tmpMatrix=self.rbf_kernel(dataToAnalyze,gamma=0.05)
        return tmpMatrix



