__author__ = 'kos'

import copy as cp

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
from datetime import date
import os
import sys
import numpy as np
import Tkinter as tk
from Tkinter import *
from tkFileDialog import *
from nhlscrapi.games.game import Game, GameKey, GameType
from utils import small_util_routines

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




