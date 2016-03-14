import org.nhl.dataAnalysis.*;
import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;

import java.util.Enumeration;

/**
 * Created by kos on 11.03.16.
 */
public class dataAnalysis {

    private static manageSQLCla createDatabase(String ending,String DBName,Instances wholeDataSet)
    {
        manageSQLCla cTS = new manageSQLCla("nhlDatabase");

        cTS.deleteAndRecreateSQLTable(DBName);
        cTS.addInstancesToSQLDatabase(wholeDataSet,DBName);

        return cTS;

    }

    public static void main(String args[]) {

        InputDataCla inputNHLDataReader = new InputDataCla("new tester class instance");
        inputNHLDataReader.getNHLFiles("_delta");
        inputNHLDataReader.convertCSVARFF();
        inputNHLDataReader.getArffFiles();
        Instances wholeDataSet=inputNHLDataReader.returnNHLData();
        manageSQLCla cTS_Delta=createDatabase("_delta","NHLDATA_DELTA",wholeDataSet);
//        inputNHLDataReader.getNHLFiles("_sum");
//        inputNHLDataReader.convertCSVARFF();
//        inputNHLDataReader.getArffFiles();
//        wholeDataSet=inputNHLDataReader.returnNHLData();
//        cTS_Delta=createDatabase("_delta","NHLDATA_SUM",wholeDataSet);

        System.out.println("now retrieve data from sql database with"+wholeDataSet.numInstances()+" instances ");

        Instances NYI_NYR_games=cTS_Delta.getInstancesFromSQL("SELECT * FROM NHLDATA_DELTA WHERE homeTeam = 'NYI' AND awayTeam = 'NYR' OR ( homeTeam = 'NYR' AND awayTeam = 'NYI' );");
        System.out.println(NYI_NYR_games.numInstances());

        Instances gamesIn2013=cTS_Delta.getInstancesFromSQL("SELECT * FROM NHLDATA_DELTA WHERE season = 2013;");
        System.out.println(gamesIn2013.numInstances());

        Instances gesamtDaten=cTS_Delta.getInstancesFromSQL("SELECT * FROM NHLDATA_DELTA;");
        System.out.println(gesamtDaten.numInstances());

        Instances datenZurAnalyse=gesamtDaten;

        splitDaten sD=new splitDaten();
        Instances datenZurAnalyseGefiltert=sD.FilterArguments("1,2,3,4,5,6","datenZurAnalyse",datenZurAnalyse);
        sD.divideInputInstancesIntoTrainTest(0.5,datenZurAnalyseGefiltert);
        sD.setClassification("score");
        Instances trainingsDaten=sD.getTrainingData();
        Instances testDaten=sD.getTestData();

        System.out.println("split delivered trainings Daten: "+trainingsDaten.numInstances());
        System.out.println("split delivered test Daten: "+testDaten.numInstances());

        klassifikation klas = new klassifikation(trainingsDaten);
        klas.setupKlassifikation();
        klas.evaluateKlassifikator(testDaten);

        sD.setClassification("shots");
        klas = new klassifikation(trainingsDaten);
        klas.setupKlassifikation();
        klas.evaluateKlassifikator(testDaten);

        System.out.println("=======================================");
        System.out.println("             cluster analysis    ");
        System.out.println("=======================================");

        clusteringCla skMCI = new clusteringCla(trainingsDaten);
        skMCI.createDatasets(0.5);
        System.out.println("Now create the clusters");
        skMCI.createCluster(3);
        skMCI.trainCluster();
        skMCI.evalCluster();
        ClusterEvaluation eval = skMCI.returnClusterEvaluator();
        int AnzahlCluster=skMCI.getAnzahlCluster();


    }

}
