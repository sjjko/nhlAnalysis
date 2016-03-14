import org.nhl.dataAnalysis.*;
import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;
import weka.gui.visualize.*;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
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

    public Boolean doInputPart=false;

    public static void main(String args[]) {

            InputDataCla inputNHLDataReader = new InputDataCla("new tester class instance");
            inputNHLDataReader.getNHLFiles("_delta");
            inputNHLDataReader.convertCSVARFF();
            inputNHLDataReader.getArffFiles();
            Instances wholeDataSet = inputNHLDataReader.returnNHLData();
        manageSQLCla cTS_Delta=createDatabase("_delta","NHLDATA_DELTA",wholeDataSet);
//        inputNHLDataReader.getNHLFiles("_sum");
//        inputNHLDataReader.convertCSVARFF();
//        inputNHLDataReader.getArffFiles();
//        wholeDataSet=inputNHLDataReader.returnNHLData();
//        cTS_Delta=createDatabase("_delta","NHLDATA_SUM",wholeDataSet);

        testingCla testEr = new testingCla(); /*do some testing on data and SQL database*/

        try {
            testEr.TestDataSet(wholeDataSet);
        } catch (testingCla.NanValueFound nanValueFound) {
            nanValueFound.printStackTrace();
        } catch (testingCla.InfiniteValueFound infiniteValueFound) {
            infiniteValueFound.printStackTrace();
        }

        System.out.println("now retrieve data from sql database with"+wholeDataSet.numInstances()+" instances ");

        Instances NYI_NYR_games=cTS_Delta.getInstancesFromSQL("SELECT * FROM NHLDATA_DELTA WHERE homeTeam = 'NYI' AND awayTeam = 'NYR' OR ( homeTeam = 'NYR' AND awayTeam = 'NYI' );");
        System.out.println(NYI_NYR_games.numInstances());

        Instances gamesIn2013=cTS_Delta.getInstancesFromSQL("SELECT * FROM NHLDATA_DELTA WHERE season = 2013;");
        System.out.println(gamesIn2013.numInstances());

        Instances gesamtDaten=cTS_Delta.getInstancesFromSQL("SELECT * FROM NHLDATA_DELTA;");
        System.out.println(gesamtDaten.numInstances());

        Instances datenZurAnalyse=gesamtDaten;

        splitDaten sD=new splitDaten();
        Instances datenZurAnalyseGefiltert=sD.FilterArguments("1,2,3,4,5,6,15","datenZurAnalyse",datenZurAnalyse);
        sD.divideInputInstancesIntoTrainTest(0.5,datenZurAnalyseGefiltert);
        sD.setClassification("score");
        Instances trainingsDaten=sD.getTrainingData();
        Instances testDaten=sD.getTestData();

        System.out.println("split delivered trainings Daten: "+trainingsDaten.numInstances());
        System.out.println("split delivered test Daten: "+testDaten.numInstances());


        System.out.println("=======================================");
        System.out.println(" regression classification analysis    ");
        System.out.println("=======================================");


        klassifikationCla klas = new klassifikationCla(trainingsDaten);
        klas.setupKlassifikationCla();
        klas.evaluateKlassifikator(testDaten);

        sD.setClassification("shots");
        trainingsDaten=sD.getTrainingData();
        testDaten=sD.getTestData();
        klas = new klassifikationCla(trainingsDaten);
        klas.setupKlassifikationCla();
        klas.evaluateKlassifikator(testDaten);

        System.out.println("=======================================");
        System.out.println("             cluster analysis    ");
        System.out.println("=======================================");

        clusteringCla skMCI = new clusteringCla(datenZurAnalyseGefiltert);
        skMCI.createDatasets(0.5);
        System.out.println("Now create the clusters");
        skMCI.createCluster(2);
        skMCI.trainCluster();
        skMCI.evalCluster();
        System.out.println("now get the 0 cluster data");
        Instances cluster0Instances = skMCI.returnClusterInstanceWithIndex(0);
        Instances cluster1Instances = skMCI.returnClusterInstanceWithIndex(1);

        System.out.println("=======================================");
        System.out.println("     plot the clustered data    ");
        System.out.println("=======================================");

        Plot2D offScreenPlot = new Plot2D();
        offScreenPlot.setSize(500,500);
        String xAxisAttributeName="score";
        String yAxisAttributeName="shots";
        String coloringIndex="score";

        int xAxisAttributeIndex=cluster0Instances.attribute(xAxisAttributeName).index();
        int yAxisAttributeIndex=cluster0Instances.attribute(yAxisAttributeName).index();
        int coloringAttributeIndex=cluster0Instances.attribute(coloringIndex).index();

        PlotData2D masterPlot = new PlotData2D(cluster0Instances);
        masterPlot.m_displayAllPoints = true;
        try {
            offScreenPlot.setMasterPlot(masterPlot);
            offScreenPlot.addPlot(new PlotData2D(cluster1Instances));

        } catch (Exception e) {
            e.printStackTrace();
        }
        //master.setPlotName("Cluster 0 Score vs. Hits");
        BufferedImage osi = new BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB);
        offScreenPlot.setXindex(xAxisAttributeIndex);
        offScreenPlot.setYindex(yAxisAttributeIndex);
        offScreenPlot.setCindex(coloringAttributeIndex);
        java.awt.Graphics g = osi.getGraphics();
        offScreenPlot.paintComponent(g);
        try {
            String ImageName = "clusterAnalysis_"+xAxisAttributeName+"_vs_"+yAxisAttributeName+".png";
            ImageIO.write(osi, "png", new File(ImageName));
        } catch (IOException e) {
            e.printStackTrace();
        }

        PNGWriter pngw=new PNGWriter();

        VisualizePanel viP=new VisualizePanel();
        viP.setName("k-means clustering of NHL data");
        viP.setShowAttBars(true);
        viP.setShowClassPanel(true);
        //viP.setFont(new Font("Ariana"));

        //ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
        try {

            PlotData2D tmpPD2D=new PlotData2D(cluster0Instances);
            boolean[] cp = new boolean[cluster0Instances.numInstances()];
            tmpPD2D.setConnectPoints(cp);
            viP.setMasterPlot(tmpPD2D);

            tmpPD2D=new PlotData2D(cluster1Instances);
            cp = new boolean[cluster1Instances.numInstances()];
            tmpPD2D.setConnectPoints(cp);
            viP.addPlot(tmpPD2D);

        } catch (Exception e) {
            e.printStackTrace();
        }

        String plotName = viP.getName();
        final javax.swing.JFrame jf =
                new javax.swing.JFrame("Weka Clusterer Visualize"+plotName);
        jf.setSize(500,500);

        jf.getContentPane().setLayout(new BorderLayout());

        jf.getContentPane().add(viP, BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
            }
        });

        jf.setVisible(true);

        System.out.println("=======================================");
        System.out.println("     finished dataAnalysis java class    ");
        System.out.println("=======================================");


    }



}
