__author__ = 'kos'

try:
    import os
    import numpy as np
    import sys
    import matplotlib.pyplot as plt
    from scipy import signal
    import math
    from utils import small_util_routines
    from datetime import date
    import os
    import sys
    import numpy as np
    import Tkinter as tk
    from Tkinter import *
    from tkFileDialog import *
    from nhlscrapi.games.game import Game, GameKey, GameType
    from utils import small_util_routines
    import copy as cp
except ImportError:
    print ImportError.message.getter
    raise ImportError('Module not found')


critical=1
nonCritical=0

GLOBAL_VERBOSITY_LEVEL=1

utC=small_util_routines(GLOBAL_VERBOSITY_LEVEL)
VerbosityF=utC.VerbosityF
ToassertF=utC.ToassertF
EnumF=utC.EnumF

"""
    this class implements harvesting and
    processing routines for NHL game data

    for fetching NHL data the nhlscrapi class
    has been used
"""

class MinerC():


    """ class for mining NHL game data"""

    def __init__(self,seasonList,withPlayoffGames):
        """initialize data structures for data scraping"""
        """
        input:
            season: 4 digit number indicating the season or string "all"
            withPlayoffGames:       True or False, if with or without postseason results
        """
        self._fileNamePrefix="dataFileNhl_"
        self._fileNamePostfix=".dat"
        self._actualSeason=0
        self._fileName=""

        """_includingPlayoffGames: true: with playoff games for that season"""
        if withPlayoffGames:
            self._includingPlayoffGames=True

        if seasonList=="all":
            self._seasonList=[2010,2011,2012,2013,2014,2015,2016]
        elif isinstance(seasonList,int):
            self._seasonList=[seasonList]
        else:
            #do nothing - empty list
            self._seasonList=[]

        if withPlayoffGames:
            self._gameTypeList=[GameType.Regular,GameType.Playoffs]
        else:
            self._gameTypeList=[GameType.Regular]

        self._gameId=0
        self._relativeDate=0
        self._homeTeam=0
        self._awayTeam=0
        self._homeWin=0
        self._score_diff=0
        self._shots_dif=0
        self._diff_fo=0
        self._deltaTA=0
        self._deltaGA=0
        self._deltaPM=0
        self._deltaHT=0
        self._deltaPM=0
        self._delta_ppgoals=0
        self._attendance=0
        self._GAMENUMAX=1200
        self._actualGameType=0


    def mine_data(self):


        """ main mining routine in mining class - fetches data using nhlscrapi """

        for season in self._seasonList:
            self._actualSeason=season
            #for every season we set the gameID to zero
            self._gameId=0
            #create files for added and delta values
            self.write_header(plusMinus=1)
            self.write_header(plusMinus=-1)
            VerbosityF(0,"Mine for data in season ",season)

            for self._actualGameType in self._gameTypeList:
                VerbosityF(1,"Fetching data for game type ",self._actualGameType)
                #number of games per season is stored in constant of NHLscrapi
                #ToassertF(C.GAME_CT_DICT[season]==0,nonCritical,"no games for season ",season)
                gameNum=0
                #for gameNum in range(self._GAMENUMAX):
                coachName="noCoach"
                while gameNum < range(self._GAMENUMAX) and not coachName=="":
                    gameNum=gameNum+1
                    game_key=GameKey(self._actualSeason,self._actualGameType,gameNum)
                    gameObj=Game(game_key)
                    try:
                        coachName=Game(GameKey(self._actualSeason,self._actualGameType,gameNum)).away_coach
                        #exist=isinstance(gameObj.plays,list)
                    except IndexError:
                        coachName=""
                    if not coachName=="":
                    #for gameNum in range(1,C.GAME_CT_DICT[season]+1):
                        VerbosityF(1,"Fetching data for game ",gameNum)
                        do_print=1
                        try:
                            gameObj.load_all()
                            dateOfMatch=gameObj.matchup["date"]
                            gameDateReformatted=self.getGameDate(dateOfMatch)
                            referenceDate=self.getReferenceDate(season)
                            self._relativeDate=(gameDateReformatted-referenceDate).days
                            sumDict=gameObj.event_summary
                            #sumDict=nhl.games.eventsummary.EventSummary(game_key)
                            #faceoff statistics
                            sumgameDict=sumDict.totals()
                            #sumgameDict=nhl.games.game.EventSummary(game_key).totals()
                            takeAwaysHome=sumgameDict["home"]["tk"]
                            takeAwaysAway=sumgameDict["away"]["tk"]
                            giveAwaysHome=sumgameDict["home"]["gv"]
                            giveAwaysAway=sumgameDict["away"]["gv"]
                            pimHome=sumgameDict["home"]["pim"]
                            pimAway=sumgameDict["away"]["pim"]
                            hitHome=sumgameDict["home"]["ht"]
                            hitAway=sumgameDict["away"]["ht"]
                            afo=sumDict.away_fo
                            hfo=sumDict.home_fo
                            hfov=hfo["ev"]["won"]
                            afov=afo["ev"]["won"]
                            score_hometeam=int(gameObj.matchup["final"]["home"])
                            score_awayteam=int(gameObj.matchup["final"]["away"])
                            #sumgame=nhl.games.game.EventSummary(game_key)
                            ppgoalsHomeTeam=sumDict.home_shots['agg']['pp']['g']
                            ppgoalsAwayTeam=sumDict.away_shots['agg']['pp']['g']
                            shotsHome=int(sumgameDict["home"]["s"])
                            shotsAway=int(sumgameDict["away"]["s"])
                            self._homeTeam=gameObj.matchup["home"]
                            self._awayTeam=gameObj.matchup["away"]
                            self._attendance=gameObj.matchup["attendance"]

                        except:
                        #do nothing - simply do not print the current dataset
                            do_print=0

    # we print two files - one with summed, one with delta values
                        for plusMinus in [1,-1]:
                            if isinstance(hfov,int):
                                self._diff_fo=int(hfov)-plusMinus*int(afov)
                            #else:
                            #    self._diff_fo="NAN"

                            self._deltaTA=takeAwaysHome-plusMinus*takeAwaysAway
                            self._deltaGA=giveAwaysHome-plusMinus*giveAwaysAway
                            self._deltaPM=pimHome-plusMinus*pimAway
                            self._deltaHT=hitHome-plusMinus*hitAway
                            self._score_diff=score_hometeam-plusMinus*score_awayteam
                            self._delta_ppgoals=ppgoalsHomeTeam-plusMinus*ppgoalsAwayTeam
                            self._shots_diff=shotsHome-plusMinus*shotsAway

                            if plusMinus==1:
                                haveWonLambda = lambda k: 1 if k>0 else 0
                                self._homeHasWon=haveWonLambda(self._score_diff)
                                self._gameId=self._gameId+1

                            #now print this dataset

                            self.write_data(do_print,plusMinus)



    def getGameDate(self,gameDate):
        """get the date of the game in the python date format"""

        dateList=str(gameDate).split(",")
        year=int(dateList[-1])
        day=int(dateList[-2].split(" ")[-1])
        monthStr=dateList[-2].split(" ")[-2]
        monthList=["January","February","March","April","May","June","July","August","September"
        ,"October","November","December"]
        monthVal = monthList.index(monthStr)+1
        dateOfGame=date(year,monthVal,day)

        return dateOfGame

    def getReferenceDate(self,season):
        """reference date to which date of every game is calculated"""
        return date(season-1,10,1)

    def write_header(self,plusMinus):
        """write output file header"""

        self.assembleFileName(self._actualSeason,plusMinus)

        try:
            f=open(self._fileName,"w+")
            f.write("ID \t season \t Date \t homeTeam \t awayTeam \t WonYN \t ScoreDiff \t ShDiff \t FacoffDiff \t TakeawayDiff \t \
            GiveawaysDiff \t PenaltyminutesDiff \t HitsDiff \t powerplayGoalsDiff \t Attendance")
            f.close()

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise


    def assembleFileName(self,thisSeason,plusMinus):
        """
        assemble name of output file
        input:
        season: 4 digit number specifying season e.g. 2010
        """
        if self._actualGameType==self._gameTypeList[0]:
            addString="_regular"
        else:
            addString="_postSeason"

        if plusMinus==1:
            self._fileName="".join([self._fileNamePrefix,str(thisSeason),addString,"_delta",self._fileNamePostfix])
        else:
            self._fileName="".join([self._fileNamePrefix,str(thisSeason),addString,"_sum",self._fileNamePostfix])


    def write_data(self,doPrint,plusMinus):
        """
        write data in ascii format into output file
        :rtype : object
        input:
        doPrint: 0 or 1, if dataset should be dumped to file
        """

        self.assembleFileName(self._actualSeason,plusMinus)

        if doPrint:

            try:
                f=open(self._fileName,"a+")
                f.write("\n")
            #ID season Date homeTeam awayTeam WonYN ScoreDiff ShDiff FacoffDiff TakeawayDiff GiveawaysDiff PenaltyminutesDiff HitsDiff powerPlayGoals Attendance


                f.write(str(self._gameId)+"\t"\
                        +str(self._actualSeason)+"\t"\
                        +str(self._relativeDate)+"\t"\
                        +str(self._homeTeam)+"\t"\
                        +str(self._awayTeam)+"\t"\
                        +str(self._homeHasWon)+"\t"\
                        +str(self._score_diff)+"\t"\
                        +str(self._shots_diff)+"\t"\
                        +str(self._diff_fo)+"\t"\
                        +str(self._deltaTA)+"\t"\
                        +str(self._deltaGA)+"\t"\
                        +str(self._deltaPM)+"\t"\
                        +str(self._deltaHT)+"\t"\
                        +str(self._delta_ppgoals)+"\t"\
                        +str(self._attendance))
                f.close()
            except IOError as e:
                print "I/O error({0}): {1}".format(e.errno, e.strerror)
            except ValueError:
                print "Could not convert data to an string."
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise

class prepareDataC():
    """prepare data for use in statistical analysis"""

    def __init__(self,seasonList,readUsingTK=True):
        """initialize local variables"""

        #self.MinerC=MinerC
        self._gesamtInputDatenDict={}
        self._gesamtInputDatenDictPostSeason={}
        self._gesamtInputDatenDictSUM={}
        self._gesamtInputDatenDictPostSeasonSUM={}

        self._seasonList=[]
        if isinstance(seasonList,int):
            self._seasonList.append(seasonList)
        elif isinstance(seasonList,list):
            for listElement in seasonList:
                self._seasonList.append(listElement)
        elif seasonList=="all":
            #look for all files in current folder and fetch the seasons
            currentPath=os.getcwd()
            allSeasonList=[2008,2009,2010,2011,2012,2013,2014,2015,2016]
            for season in allSeasonList:
                filename="".join(currentPath+"/dataSetsNHL/dataFileNhl_"+str(season)+"_regular_delta.dat")
                #now check if file exists
                VerbosityF(2,"check if file ",filename," exists!")
                if os.path.isfile(filename):
                    VerbosityF(2,"file ",filename," exists!")
                    self._seasonList.append(season)
                else:
                    VerbosityF(2,"file ",filename," NOT FOUND in current directory!")

        self._gesamtDatenDict={}
        self._gesamtDatenDictPostSeason={}
        self._gesamtDatenDictSUM={}
        self._gesamtDatenDictPostSeasonSUM={}

        self._gesamtDatenGeneratedCorrelated={}
        self._gesamtDatenGeneratedUnCorrelated={}
        self._gesamtDatenGeneratedUniformUncorrelated={}

        #now read the data as given in the input
        file_path=""
        if readUsingTK:
        #now read using an open dialog
            root = tk.Tk()
            root.withdraw()
            file_path = askopenfilename()

        #file_path is empty if not read by TK
        #else the filename is guessed by using the internal
        #list of seasons in self._seasonList
        self.readInput(file_path,readUsingTK)

        #self._gesamtInputDatenDict=gesamtInputDatenDict

        self._EN=EnumF("ID","season","date","homeTeam","awayTeam","wonYN","ScoreDiff","ShDiff","FaceoffDiff","TakeawayDiff", \
     "GiveawaysDiff","PenaltyminutesDiff","HitsDiff","powerplayGoalsDiff","Attendance")

    #    self._ENList=["ID","season","date","homeTeam","awayTeam","wonYN","ScoreDiff","ShDiff","FaceoffDiff","TakeawayDiff", \
    # "GiveawaysDiff","PenaltyminutesDiff","HitsDiff","powerplayGoalsDiff","Attendance"]

        self._ENList=["ID","season","date","homeTeam","awayTeam","wonYN","score","shots","faceoffs","takeaways", \
     "giveaways","penaltyminutes","hits","powerplay goals","attendance"]

        #print self._EN.ID,self._EN.season,self._EN.date

        self._DataSetType=EnumF("delta","summed","correlated","uncorrelated_normal","uncorrelated_uniform")
        self._DataSetTypeList=["regular season data - differences","regular season data - summed","correlated","uncorrelated_normal","uncorrelated_uniform"]


    def ChopOffLastList(self,inputData):
        return inputData[:-1]


    def prepareData(self):

        for keyString in sorted(self._gesamtInputDatenDict.keys()):
            #now prepare the data
            tmpData=self.prepareSeasonData(self._gesamtInputDatenDict[keyString])
            self._gesamtDatenDict[keyString]=self.ChopOffLastList(tmpData)
            tmpData=self.prepareSeasonData(self._gesamtInputDatenDictSUM[keyString])
            self._gesamtDatenDictSUM[keyString]=self.ChopOffLastList(tmpData)
            if not self._gesamtInputDatenDictPostSeason=={}:
                self._gesamtDatenDictPostSeason[keyString]=self.prepareSeasonData(self._gesamtInputDatenDictPostSeason[keyString])
            else:
                VerbosityF(1,"no postseason delta data found")
            if not self._gesamtInputDatenDictPostSeason=={}:
                self._gesamtDatenDictPostSeasonSUM[keyString]=self.prepareSeasonData(self._gesamtInputDatenDictPostSeasonSUM[keyString])
            else:
                VerbosityF(1,"no postseason sum data found")

        VerbosityF(0,"data preparation finished.")
        VerbosityF(0,"we have found, read and processed the following datasets:")
        VerbosityF(0,"following seasons for delta values: \n",sorted(self._gesamtDatenDict.keys()))
        VerbosityF(0,"following seasons for summed values: \n",sorted(self._gesamtDatenDictSUM.keys()))


    def returnDataForSeason(self,season):
        return self._gesamtDatenDict[season]

    def prepareSeasonData(self,inputData):

        #""
        #"input: block of input data, in string format
        #"output: Nxk numpy output array of data, with the numerical part converted to numerical format
        #""

        captionListe=inputData.split("\n")[0]
        VerbosityF(2,"input data contains the following k-datasets as columns:")
        VerbosityF(2,"".join(captionListe))
        zeilenDatenListe=inputData.split("\n")[1:]
        datenZeilenSpalten=[p.split("\t") for p in zeilenDatenListe]
        return [ [int(k) if k.replace('-','').isdigit() else k for k in p ] for p in datenZeilenSpalten ]

    def enum(*sequential, **named):
        enums = dict(zip(sequential, range(len(sequential))), **named)
        return type('Enum', (), enums)


    def readInput(self,file_path,readUsingTK=False):
        """read raw data from file and put it in dictionary"""
        try:
            if file_path=="":
                for season in self._seasonList:
                    #now read the season and postseason data - differences and added values
                    addOrDeltaList=["_sum","_delta"]
                    regularOrPostseason=["_regular","_postSeason"]
                    for addOrDelta in addOrDeltaList:
                        for regularOrPlayoff in regularOrPostseason:
                            currentPath=os.getcwd()
                            filename="".join(currentPath+"/dataSetsNHL/dataFileNhl_"+str(season)+regularOrPlayoff+addOrDelta+".dat")
                            VerbosityF(1,"now read file",filename)
                            try:
                                f=open(filename,"r")
                                ToassertF(isinstance(file_path,str),critical,"filepath ",file_path, " is not a string!")
                                season=int(filename.split("_")[1].split(".")[0])
                                #use the dictionary structure for convience and readability
                                if addOrDelta=="_delta":
                                    if regularOrPlayoff=="_regular":
                                        self._gesamtInputDatenDict[season]=f.read()
                                    elif regularOrPlayoff=="_postSeason":
                                        self._gesamtInputDatenDictPostSeason[season]=f.read()
                                elif addOrDelta=="_sum":
                                    if regularOrPlayoff=="_regular":
                                        self._gesamtInputDatenDictSUM[season]=f.read()
                                    elif regularOrPlayoff=="_postSeason":
                                        self._gesamtInputDatenDictPostSeasonSUM[season]=f.read()
                            except IOError:
                                VerbosityF(1,"file with filename ",filename," not found!")
                            f.close()
            else:
                f=open(file_path,"r")
                ToassertF(isinstance(file_path,str),critical,"filepath ",file_path, " is not a string!")
                season=int(file_path.split("_")[-1].split(".")[0])
                ToassertF(isinstance(season,int),critical,"season as extracted is ",season, " and is not a number!")
                self._gesamtInputDatenDict[season]=f.read()
                f.close()

        except:
            print "could not read input data for season ", season

    def GenerateGenericDistributedDataUnCorrelated(self):
        VerbosityF(0,"generate unrelated normal distributed multivariate data")
        unitCovDict={}
        #first write the delta data
        self._gesamtDatenGeneratedUnCorrelated=self.GenerateGenericDistributedDataHelper(self._gesamtDatenDict,unitCovDict,True,uniformDistribution=False)
        #now write the sum file
        self._gesamtDatenGeneratedUnCorrelated=self.GenerateGenericDistributedDataHelper(self._gesamtDatenDictSUM,unitCovDict,True,uniformDistribution=False)

    def GenerateGenericDistributedDataCorrelated(self,unitCovDict):

        VerbosityF(0,"generate normal distributed multivariate data with predefined crosscorrelation")
        self._gesamtDatenGeneratedCorrelated=self.GenerateGenericDistributedDataHelper(self._gesamtDatenDict,unitCovDict,False,uniformDistribution=False)
        self._gesamtDatenGeneratedCorrelated=self.GenerateGenericDistributedDataHelper(self._gesamtDatenDictSUM,unitCovDict,False,uniformDistribution=False)

    def GenerateGenericDistributedDataUniform(self):

        VerbosityF(0,"generate uniformly distributed multivariate data")
        unitCovDict={}
        self._gesamtDatenGeneratedUniformUncorrelated=self.GenerateGenericDistributedDataHelper(self._gesamtDatenDict,unitCovDict,False,uniformDistribution=True)
        self._gesamtDatenGeneratedUniformUncorrelated=self.GenerateGenericDistributedDataHelper(self._gesamtDatenDictSUM,unitCovDict,False,uniformDistribution=True)



    def GenerateGenericDistributedDataHelper(self,originalInputData,unitCovDictInput,uncorrelated,uniformDistribution=False):

        """
        to generate correlated output data according to a multivariate normal distribution
        for testing reasons
        input:
            fullSeasonData a realistic NHL dataset - keep all games
                            just alter the results to correlate
        """

        captionList=["ScoreDiff","shotsDiff","FaceoffDiff",\
                     "TakeawayDiff","GiveawaysDiff","PenaltyminutesDiff",\
                     "HitsDiff","powerplayGoalsDiff"]

        unitCovDict={}
        for cap1 in captionList:
            for cap2 in captionList:
                #initialize completely uncorrelated
                unitCovDict[cap1,cap2]=0

        #then add some correlations - between 0 and 1:

        if unitCovDictInput=={}:
            VerbosityF(0,"no cross correlations set between datasets")
            #if no covariances are prefixed - do the standard ones
            unitCovDict["ScoreDiff","shotsDiff"]=0.5
            unitCovDict["FaceoffDiff","GiveawaysDiff"]=0.7
            unitCovDict["HitsDiff","PenaltyminutesDiff"]=0.6
            #6-8
            unitCovDict["PenaltyminutesDiff","powerplayGoalsDiff"]=1

        if uncorrelated==True:
            VerbosityF(0,"generate uncorrelated data - independent normal distributions around a fixed mean and fixed variance")

        #now loop over all the seasons
        for season in originalInputData.keys():

            wonYN=[p[self._EN.wonYN] for p in originalInputData[season]]
            ScoreDiff=[p[self._EN.ScoreDiff] for p in originalInputData[season]]
            shotsDiff=[p[self._EN.ShDiff] for p in originalInputData[season]]
            FaceoffDiff=[p[self._EN.FaceoffDiff] for p in originalInputData[season]]
            TakeawayDiff=[p[self._EN.TakeawayDiff] for p in originalInputData[season]]
            GiveawaysDiff=[p[self._EN.GiveawaysDiff] for p in originalInputData[season]]
            PenaltyminutesDiff=[p[self._EN.PenaltyminutesDiff] for p in originalInputData[season]]
            HitsDiff=[p[self._EN.HitsDiff] for p in originalInputData[season]]
            powerplayGoalsDiff=[p[self._EN.powerplayGoalsDiff] for p in originalInputData[season]]

            #now generate a multivariate distribution#

            mean=[np.mean(ScoreDiff)]
            mean=mean+[np.mean(shotsDiff)]
            mean=mean+[np.mean(FaceoffDiff)]
            mean=mean+[np.mean(TakeawayDiff)]
            mean=mean+[np.mean(GiveawaysDiff)]
            mean=mean+[np.mean(PenaltyminutesDiff)]
            mean=mean+[np.mean(HitsDiff)]
            mean=mean+[np.mean(powerplayGoalsDiff)]

            covii=[np.cov(ScoreDiff)]
            covii=covii+[np.cov(shotsDiff)]
            covii=covii+[np.cov(FaceoffDiff)]
            covii=covii+[np.cov(TakeawayDiff)]
            covii=covii+[np.cov(GiveawaysDiff)]
            covii=covii+[np.cov(PenaltyminutesDiff)]
            covii=covii+[np.cov(HitsDiff)]
            covii=covii+[np.cov(powerplayGoalsDiff)]


            cov=np.zeros([len(mean),len(mean)])
            for i in range(len(mean)):
             for j in range(len(mean)):
                 if i==j:
                    cov[i,j]=covii[i]
                 else:
                     #correlations between 0 and 1 in units of self correlation coefficients
                    cov[i,j]=covii[i]*covii[j]*unitCovDict[captionList[i],captionList[j]]

            #mean = np.asarray([0, 0])
            #cov = np.asarray([[100, 77], [0, 50]])
            numberOfGeneratedGames=len(wonYN)
            numberOfGeneratedCategories=len(captionList)

            xout=np.zeros([numberOfGeneratedCategories,numberOfGeneratedGames])

            if not uniformDistribution:
                xoutSum = np.random.multivariate_normal(mean, cov, numberOfGeneratedGames).T
                xoutDelta = np.random.multivariate_normal(mean, cov, numberOfGeneratedGames).T

            else:
                VerbosityF(0,"now we generate uniform distributed data")
                for i in range(numberOfGeneratedCategories):
                    #if writeDeltaDataTrue==False:
                    #    minimum=1
                    #else:
                    #    minimum=int(mean[i]-2*np.sqrt(cov[i,i]))
                    maximum=int(mean[i]+2*np.sqrt(cov[i,i]))
                    minimum=-max(-minimum,maximum)
                    maximum=abs(minimum)
                    VerbosityF(1,"minimum for distribution set to ",minimum," maximum is ",maximum," with mean ",mean[i])
                    xout[i,:]=np.random.random_integers(minimum, high=maximum, size=len(xout[i,:]))


            #now transform variables to realistic values

            #first transform all the values to int
            for i in range(min(xout.shape)):
                for j in range(max(xout.shape)):
                    xout[i,j]=int(xout[i,j])
                    #cut off maximal values at an arbitrary value of +-20
                    if abs(xout[i,j])>20:
                        xout[i,j]=np.sign(xout[i,j])*19


            #for summed values - all entries have to be positive
            #if writeDeltaDataTrue==False:
            #    for i in range(min(xout.shape)):
            #        for j in range(max(xout.shape)):
            #            if xout[i,j]<0:
            #                xout[i,j]=-xout[i,j]


            wonYNList=[]
            #if home team scores more goals it has won
            for i in range(max(xout.shape)):
                scoreDiff=xout[0,i]
                if scoreDiff>0.5:
                    #0 is category of wins
                    wonYNList.append(1)
                else:
                    wonYNList.append(0)


            #now transform the data back into the input array
            #fullSeasonArray=np.asarray(fullSeasonData)

            generatedDataDict={}
            #for seasonkey in originalInputData.keys():

                #if uncorrelated==True and uniformDistribution==False:
                #elif uncorrelated==False and uniformDistribution==False:
                #elif uniformDistribution==True:

            #the first five kategories
            #"ID","season","date","homeTeam","awayTeam"
            # are not altered
            writeDeltaDataTrue=False
            actualGameType=GameType.Regular
            #get the filename for the generated data
            #fileName=self.assembleGeneratedFileName(season,actualGameType,writeDeltaDataTrue,uncorrelated,uniformDistribution)
            #first overwrite existing files with same filename#
            #f=open(fileName,"w+")
            #f.write("\n")
            #f.close()

            generatedDataDict[season]=[]
            for gameID in range(numberOfGeneratedGames):
                #we make a shallow copy of the orginial data - replacing only the generated data
                rowToAppend=cp.copy(originalInputData[season][gameID])
                #but now alter entries 5 to last one
                rowToAppend[self._EN.wonYN]=wonYNList[gameID]
                rowToAppend[self._EN.ScoreDiff]=xout[0,gameID]
                rowToAppend[self._EN.ShDiff]=xout[1,gameID]
                rowToAppend[self._EN.FaceoffDiff]=xout[2,gameID]
                rowToAppend[self._EN.TakeawayDiff]=xout[3,gameID]
                rowToAppend[self._EN.GiveawaysDiff]=xout[4,gameID]
                rowToAppend[self._EN.PenaltyminutesDiff]=xout[5,gameID]
                rowToAppend[self._EN.HitsDiff]=xout[6,gameID]
                rowToAppend[self._EN.powerplayGoalsDiff]=xout[7,gameID]

                #now append the data again
                generatedDataDict[season]=generatedDataDict[season]+[rowToAppend]

                try:
                    f=open(fileName,"a+")
                    f.write("\n")
                #ID season Date homeTeam awayTeam WonYN ScoreDiff ShDiff FacoffDiff TakeawayDiff GiveawaysDiff PenaltyminutesDiff HitsDiff powerPlayGoals Attendance

                    f.write(str(rowToAppend[self._EN.ID])+"\t"\
                            +str(rowToAppend[self._EN.season])+"\t"\
                            +str(rowToAppend[self._EN.date])+"\t"\
                            +str(rowToAppend[self._EN.homeTeam])+"\t"\
                            +str(rowToAppend[self._EN.awayTeam])+"\t"\
                            +str(rowToAppend[self._EN.wonYN])+"\t"\
                            +str(rowToAppend[self._EN.ScoreDiff])+"\t"\
                            +str(rowToAppend[self._EN.ShDiff])+"\t"\
                            +str(rowToAppend[self._EN.FaceoffDiff])+"\t"\
                            +str(rowToAppend[self._EN.TakeawayDiff])+"\t"\
                            +str(rowToAppend[self._EN.GiveawaysDiff])+"\t"\
                            +str(rowToAppend[self._EN.PenaltyminutesDiff])+"\t"\
                            +str(rowToAppend[self._EN.HitsDiff])+"\t"\
                            +str(rowToAppend[self._EN.powerplayGoalsDiff])+"\t"\
                            +str(rowToAppend[self._EN.Attendance])+"\t"\
                            +str(rowToAppend[self._EN.ID]))
                    f.close()
                except IOError as e:
                    print "I/O error({0}): {1}".format(e.errno, e.strerror)
                except ValueError:
                    print "Could not convert data to an string."
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise


        return generatedDataDict


    def assembleGeneratedFileName(self,thisSeason,actualGameType,plusMinus,uncorrelated,uniformDistribution):
        """
        assemble name of output file
        input:
        season: 4 digit number specifying season e.g. 2010
        """
        if actualGameType==1:#self.MinerC._gameTypeList[0]:
            addString="_regular"
        else:
            addString="_postSeason"

        if uncorrelated and not uniformDistribution:
            addStringCorrelationMethod="_uniform"
        elif  uncorrelated and uniformDistribution:
            addStringCorrelationMethod="_normal_uncorrelated"
        else:
            addStringCorrelationMethod="_normal_correlated"

        if plusMinus==1:
            fileName="".join([self.MinerC._fileNamePrefix,str(thisSeason),addString,addStringCorrelationMethod,"_delta",self.MinerC._fileNamePostfix])
        else:
            fileName="".join([self.MinerC._fileNamePrefix,str(thisSeason),addString,addStringCorrelationMethod,"_sum",self.MinerC._fileNamePostfix])

        return fileName

    def GenerateGenericDataUnCorrelated(self):

        VerbosityF(0,"generate unrelated normal distributed multivariate data")
        unitCovDict={}
        self._gesamtDatenGeneratedUnCorrelated=self.GenerateGenericDataCorrelatedHelper(unitCovDict,True,False)

    def GenerateGenericDataCorrelated(self,unitCovDict):

        VerbosityF(0,"generate normal distributed multivariate data with predefined crosscorrelation")
        self._gesamtDatenGeneratedCorrelated=self.GenerateGenericDataCorrelatedHelper(unitCovDict,False,False)

    def GenerateGenericDataUniform(self):

        VerbosityF(0,"generate uniformly distributed multivariate data")
        unitCovDict={}
        self._gesamtDatenGeneratedUniformUncorrelated=self.GenerateGenericDataCorrelatedHelper(unitCovDict,False,uniformDistribution=True)


    def GenerateGenericDataCorrelatedHelper(self,unitCovDictInput,uncorrelated,uniformDistribution=False):

        """
        to generate correlated output data according to a multivariate normal distribution
        for testing reasons
        input:
            fullSeasonData a realistic NHL dataset - keep all games
                            just alter the results to correlate
        """

        #tmpPlaceHolder=np.zeros(NoGames)
        #fullGenericDataSet=np.array(fullSeasonData)

        #for id in NoGames:
        #    for team in teams:
        #            x=0

        captionList=["ScoreDiff","shotsDiff","FaceoffDiff",\
                     "TakeawayDiff","GiveawaysDiff","PenaltyminutesDiff",\
                     "HitsDiff","powerplayGoalsDiff"]


        unitCovDict={}
        for cap1 in captionList:
            for cap2 in captionList:
                #initialize completely uncorrelated
                unitCovDict[cap1,cap2]=0

        #then add some correlations - between 0 and 1:

        if unitCovDictInput=={}:
            VerbosityF(0,"no cross correlations set between datasets")
            #if no covariances are prefixed - do the standard ones
            unitCovDict["ScoreDiff","shotsDiff"]=0.5
            unitCovDict["FaceoffDiff","GiveawaysDiff"]=0.7
            unitCovDict["HitsDiff","PenaltyminutesDiff"]=0.6
            #6-8
            unitCovDict["PenaltyminutesDiff","powerplayGoalsDiff"]=1

        if uncorrelated==True:
            VerbosityF(0,"generate uncorrelated data - independent normal distributions around a fixed mean and fixed variance")

        #now loop over all the seasons
        for season in self._gesamtDatenDict.keys():

            wonYN=[p[self._EN.wonYN] for p in self._gesamtDatenDict[season]]
            ScoreDiff=[p[self._EN.ScoreDiff] for p in self._gesamtDatenDict[season]]
            shotsDiff=[p[self._EN.ShDiff] for p in self._gesamtDatenDict[season]]
            FaceoffDiff=[p[self._EN.FaceoffDiff] for p in self._gesamtDatenDict[season]]
            TakeawayDiff=[p[self._EN.TakeawayDiff] for p in self._gesamtDatenDict[season]]
            GiveawaysDiff=[p[self._EN.GiveawaysDiff] for p in self._gesamtDatenDict[season]]
            PenaltyminutesDiff=[p[self._EN.PenaltyminutesDiff] for p in self._gesamtDatenDict[season]]
            HitsDiff=[p[self._EN.HitsDiff] for p in self._gesamtDatenDict[season]]
            powerplayGoalsDiff=[p[self._EN.powerplayGoalsDiff] for p in self._gesamtDatenDict[season]]

            #now generate a multivariate distribution#

            mean=[np.mean(ScoreDiff)]
            mean=mean+[np.mean(shotsDiff)]
            mean=mean+[np.mean(FaceoffDiff)]
            mean=mean+[np.mean(TakeawayDiff)]
            mean=mean+[np.mean(GiveawaysDiff)]
            mean=mean+[np.mean(PenaltyminutesDiff)]
            mean=mean+[np.mean(HitsDiff)]
            mean=mean+[np.mean(powerplayGoalsDiff)]

            covii=[np.cov(ScoreDiff)]
            covii=covii+[np.cov(shotsDiff)]
            covii=covii+[np.cov(FaceoffDiff)]
            covii=covii+[np.cov(TakeawayDiff)]
            covii=covii+[np.cov(GiveawaysDiff)]
            covii=covii+[np.cov(PenaltyminutesDiff)]
            covii=covii+[np.cov(HitsDiff)]
            covii=covii+[np.cov(powerplayGoalsDiff)]


            cov=np.zeros([len(mean),len(mean)])
            for i in range(len(mean)):
             for j in range(len(mean)):
                 if i==j:
                    cov[i,j]=covii[i]
                 else:
                     #correlations between 0 and 1 in units of self correlation coefficients
                    cov[i,j]=covii[i]*covii[j]*unitCovDict[captionList[i],captionList[j]]


            #mean = np.asarray([0, 0])
            #cov = np.asarray([[100, 77], [0, 50]])
            numberOfGeneratedGames=len(wonYN)
            numberOfGeneratedCategories=len(captionList)


            xout=np.zeros([numberOfGeneratedCategories,numberOfGeneratedGames])

            if not uniformDistribution:
                xout = np.random.multivariate_normal(mean, cov, numberOfGeneratedGames).T
            else:
                VerbosityF(0,"now we generate uniform distributed data")
                for i in range(numberOfGeneratedCategories):
                    minimum=int(mean[i]-2*np.sqrt(cov[i,i]))
                    maximum=int(mean[i]+2*np.sqrt(cov[i,i]))
                    minimum=-max(-minimum,maximum)
                    maximum=abs(minimum)
                    VerbosityF(1,"minimum for distribution set to ",minimum," maximum is ",maximum," with mean ",mean[i])
                    xout[i,:]=np.random.random_integers(minimum, high=maximum, size=len(xout[i,:]))


            #now transform variables to realistic values

            #first transform all the values to int
            for i in range(min(xout.shape)):
                for j in range(max(xout.shape)):
                    xout[i,j]=int(xout[i,j])
                    #cut off maximal values at an arbitrary value of +-20
                    if abs(xout[i,j])>20:
                        xout[i,j]=np.sign(xout[i,j])*19

            wonYNList=[]
            #if home team scores more goals it has won
            for i in range(max(xout.shape)):
                scoreDiff=xout[0,i]
                if scoreDiff>0.5:
                    #0 is category of wins
                    wonYNList.append(1)
                else:
                    wonYNList.append(0)


            #now transform the data back into the input array
            #fullSeasonArray=np.asarray(fullSeasonData)

            generatedDataDict={}
            for seasonkey in self._gesamtDatenDict.keys():

                #if uncorrelated==True and uniformDistribution==False:
                #elif uncorrelated==False and uniformDistribution==False:
                #elif uniformDistribution==True:



            #the first five kategories
            #"ID","season","date","homeTeam","awayTeam"
            # are not altered
                generatedDataDict[seasonkey]=[]
                for gameID in range(numberOfGeneratedGames):
                    #we make a shallow copy
                    rowToAppend=cp.copy(self._gesamtDatenDict[season][gameID])
                    #but now alter entries 5 to last one
                    rowToAppend[self._EN.wonYN]=wonYNList[gameID]
                    rowToAppend[self._EN.ScoreDiff]=xout[0,gameID]
                    rowToAppend[self._EN.ShDiff]=xout[1,gameID]
                    rowToAppend[self._EN.FaceoffDiff]=xout[2,gameID]
                    rowToAppend[self._EN.TakeawayDiff]=xout[3,gameID]
                    rowToAppend[self._EN.GiveawaysDiff]=xout[4,gameID]
                    rowToAppend[self._EN.PenaltyminutesDiff]=xout[5,gameID]
                    rowToAppend[self._EN.HitsDiff]=xout[6,gameID]
                    rowToAppend[self._EN.powerplayGoalsDiff]=xout[7,gameID]

                    generatedDataDict[seasonkey]=generatedDataDict[seasonkey]+[rowToAppend]

                #for kategoryID in range(numberOfGeneratedCategories):
                #    #now add all the data into the final array
                #    generatedDataDict[seasonkey]=generatedDataDict[seasonkey]+xout[kategoryID,:]


            return generatedDataDict

            id=0
            for i in range(min(xout.shape)):
                for j in range(min(xout.shape)):
                    if i==0 and j==1 \
                    or i==1 and j==2 \
                    or i==3 and j==5 \
                    or i==7 and j==6 \
                    or i==6 and j==8:
                        id=id+1
                        Label="".join(captionList[i]+"-"+captionList[j])
                        x=xout[i,:]
                        y=xout[j,:]
                        plt.subplot(3,2,id)
                        ax=plt.plot(x, y, 'o',label=Label)
                        plt.legend()
                        #axes = plt.gca()
                        #axes.set_ylim([-0.5,1.5])


            plt.show()

            #class names start uppercase, end with upper C

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

        VerbosityF(0,"SetupNeuralNetwork: add ",NInputLayerNeurons," layers for input")
        VerbosityF(0,"SetupNeuralNetwork: add ",NhiddenLayers," hidden layers ")
        VerbosityF(0,"SetupNeuralNetwork: add ",NHiddenLayerNeurons," hidden layer neurons ")
        VerbosityF(0,"SetupNeuralNetwork: add ",NOutputLayerNeurons," output neurons ")

        hiddenLayerList=[]
        for i in range(NhiddenLayers):
            hiddenLayerList.append(SigmoidLayer(NHiddenLayerNeurons))
        outLayer = LinearLayer(NOutputLayerNeurons)


        #now add the layers to the network
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
            VerbosityF(0,"add hidden connection: add ",List_hidden1_to_hidden2[-1])
        hidden_to_out = FullConnection(hiddenLayerList[-1], outLayer)
        #now add the connections to the network
        self._nn.addConnection(in_to_hidden)
        #add all the hidden layers
        for hidden_to_hidden_conn in List_hidden1_to_hidden2:
            self._nn.addConnection(hidden_to_hidden_conn)
        self._nn.addConnection(hidden_to_out)
        self._nn.sortModules()

        VerbosityF(0,"Finished setup of a feed forward neural network with ",len(hiddenLayerList)," hidden layers and ",NHiddenLayerNeurons," neurons in the hidden layers.")


    def PlotNeuralNetwork(self):

        """plot the neural network using the NetworkX framework"""

        #print "all modules ",self._nn.modules
        #print "in modules: ",self._nn.inmodules
        #print "out modules: ",self._nn.outmodules
        #print "between modules: ",self._nn.sortModules()
        #print self._nn.modulesSorted

        NModules=0
        for mod in self._nn.modulesSorted:
            #print mod
            NModules=NModules+1
            # #print "self._nn.connections[mod]: ",self._nn.connections[mod]
            # for conn in self._nn.connections[mod]:
            #     for cc in range(len(conn.params)):
            #         print mod,conn.whichBuffers(cc),NModules

        #print "======"

        modI=0
        indexHidden=1
        NetworkDict={}
        listNodes=[]
        listNodesLastLayer=[]
        for mod in self._nn.modulesSorted:
            conn=self._nn.connections[mod]
            listNodes=[]
            for conn in self._nn.connections[mod]:
                #numberOfNodes=len(conn.whichBuffers(cc)[0])

                for cc in range(len(conn.params)):
                     #print conn.whichBuffers(cc), conn.params[cc]
                    if modI==NModules-2:
                        listNodesLastLayer.append(conn.whichBuffers(cc)[1])

                    if modI<NModules-1:
                        listNodes.append(conn.whichBuffers(cc)[0])
                    else:
                        listNodes=listNodesLastLayer

            #print "module ",mod," has ",len(set(listNodes))," nodes !",NModules
            #get number of nodes for every slice
            if modI==0:
                keyVal="I"  #set key for the intro layer
                #print "found I"
            elif modI<NModules-1:
                keyVal="H"+str(indexHidden) #set key for the hidden layer
                indexHidden=indexHidden+1
                #print "found ",keyVal
            elif modI==NModules-1:
                keyVal="O" #the output layer
                #print "found ", keyVal, "with ",len(set(listNodesLastLayer)),"layers"
                listNodes=listNodesLastLayer

            NetworkDict[keyVal]=len(set(listNodes))
            modI=modI+1


        import networkx as nx

        xindex=0 #the index for the leftmost grid point
        pos={}
        spreadfactor=5.0

        for keyvals in NetworkDict.keys():
            noNodes=NetworkDict[keyvals]
            yindex=-spreadfactor*round(noNodes/2.)
            for i in range(0,noNodes):
                keyName=keyvals+str(i)
                #print "added node ",pos[keyName]," keyName ",keyName
                yindex=yindex+spreadfactor
                pos[keyName]=(xindex,yindex)
            xindex=xindex+1

        #pos = {"I1": (0,1),"I2": (0,-1), "H1": (1,-1),"H2": (1,0), "H3": (1,1)}
        #pos ["O1"]=(2,0)
        X=nx.Graph()
        X.add_nodes_from(pos.keys())
        nx.draw(X, pos)

        #G=nx.Graph()
        #G.add_nodes_from(["I1","I2","H1","H2","H3","O1"])

        for i in range(len(NetworkDict.keys())-1):
            for j in range(NetworkDict[NetworkDict.keys()[i]]):
                nodeName1=NetworkDict.keys()[i]+str(j)
                for k in range(NetworkDict[NetworkDict.keys()[i+1]]):
                    nodeName2=NetworkDict.keys()[i+1]+str(k)
                    X.add_edge(nodeName1,nodeName2)

        nx.draw_networkx_nodes(X,pos,node_size=200)
        nx.draw_networkx_edges(X,pos,width=1)
        nx.draw_networkx_labels(X,pos,font_size=3,font_family='sans-serif')
        plt.show() # display

        # G=nx.house_graph()
        # # explicitly set positions
        # pos={0:(0,0),
        #      1:(1,0),
        #      2:(0,1),
        #      3:(1,1),
        #      4:(0.5,2.0)}
        #
        # nx.draw_networkx_nodes(G,pos,node_size=2000,nodelist=[4])
        # nx.draw_networkx_nodes(G,pos,node_size=3000,nodelist=[0,1,2,3],node_color='b')
        # nx.draw_networkx_edges(G,pos,alpha=0.5,width=6)



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




