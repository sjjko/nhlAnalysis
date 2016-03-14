
import com.webfirmframework.wffweb.css.CssColorName;
import org.jdmp.core.algorithm.basic.Systemtime;
import org.jdmp.core.dataset.ListDataSet;
import org.nhl.dataAnalysis.*;
import org.rendersnake.HtmlCanvas;
import org.rendersnake.ext.jquery.JQueryLibrary;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import weka.gui.visualize.MatrixPanel;
import weka.gui.visualize.Plot2D;
import weka.gui.visualize.PlotData2D;

import javax.imageio.ImageIO;
import javax.swing.*;

import static org.rendersnake.HtmlAttributesFactory.class_;
import static org.rendersnake.HtmlAttributesFactory.src;
import static weka.core.Utils.getOption;

import static j2html.TagCreator.*;

//import static com.webfirmframework.wffweb.*;

/**
 * Created by kos on 02.03.16.
 */


public class Demo {
    public static void main(String args[]) {
        // Create an instance of




        //System.exit(-1);


        InputDataCla testInstance = new InputDataCla("hello");

        testInstance.getNHLFiles("_delta");
        //testInstance.getNHLFiles("_sum");

        testInstance.convertCSVARFF();
        testInstance.getArffFiles();
        //Instances wholeDataSet=testInstance.getDataAndConcatenateSources();
        System.out.println("Now prepare the test data");

        String[] filterOptions = new String[2];
        filterOptions[0] = "-R";                                    // "range"
        filterOptions[1] = "1,2,3,4,5,6,15";
        testInstance.writeMainDataInstance(testInstance.FilterArguments(filterOptions,"mainData"));
        filterOptions[0] = "-R";                                    // "range"
        filterOptions[1] = "7,8,9,10,11,12,13,14,15";
        testInstance.writeHeaderDataInstance(testInstance.FilterArguments(filterOptions,"headerData"));
        Instances mainData=testInstance.getMainDataInstances();
        testInstance.divideDataTrainTest(0.5);
        Instances trainingsDaten=testInstance.getMainTrainDataInstances();
        Instances testDaten=testInstance.getMainTestDataInstances();
        System.out.println(trainingsDaten.numAttributes());
        System.out.println(trainingsDaten.numInstances());
        System.out.println(trainingsDaten.attribute(0).name());
        mainData.setClassIndex(0);

        int indexToClassify=0;//testDaten.attribute("SCORE").index();
//        System.out.println("We classify for attribute " + testDaten.attribute("SCORE").name() + " with index " + indexToClassify);
        trainingsDaten.setClassIndex(indexToClassify); // classify data for scoring
        testDaten.setClassIndex(indexToClassify); // classify data for scoring

        manageSQLCla cTS = new manageSQLCla("nhlDatabase");
        System.out.println("we write trainingsdata to sql database: " + trainingsDaten.numInstances());
        System.out.println("write the whole dataset contained in all arff files found into database");


/*        System.out.println(wholeDataSet.numInstances());
        System.out.println(wholeDataSet.numAttributes());
        System.out.println(wholeDataSet.checkForAttributeType(4));

        cTS.deleteAndRecreateSQLTable("NHLDATA_DELTA");
        cTS.addInstancesToSQLDatabase(wholeDataSet,"NHLDATA_DELTA");
        Instances testDaten2015=cTS.getInstancesFromSQL("SELECT * FROM NHLDATA_DELTA WHERE homeTeam='NYI'");
        System.out.println("We have found in 2015 " + testDaten2015.numInstances() + " instances");
        System.out.println("We have found " + trainingsDaten.numInstances() + " instances overall");
        System.exit(-1);*/

//        DatabaseLoader loader = new DatabaseLoader();
//        loader.setSource("jdbc:mysql://localhost:3306/cuaca","root","491754");
//        loader.setQuery("select * from data_training");
//        Instances data = loader.getDataSet();
//        data.setClassIndex(data.numAttributes() - 1);

//        klassifikation klas = new klassifikation(trainingsDaten);
//
//        klas.setupKlassifikation();
//        //System.exit(-1);
//
//        klas.evaluateKlassifikator(testDaten);
        System.exit(-1);


        clusteringCla skMCI = new clusteringCla(mainData);
        skMCI.createDatasets(0.5);
        System.out.println("Now create the clusters");
        skMCI.createCluster(3);
        skMCI.trainCluster();
        skMCI.evalCluster();
        ClusterEvaluation eval = skMCI.returnClusterEvaluator();
        int AnzahlCluster=skMCI.getAnzahlCluster();

        System.out.println("Now we do the cluster analysis");


        Instances testData=skMCI.getTestInstance();
        double[] cA = eval.getClusterAssignments();
        int[] cI = eval.getClassesToClusters();
        int[] clusterSize=new int[AnzahlCluster];
        ArrayList scoreList = new ArrayList();
        Instances Cluster0= new Instances(testData,0),Cluster1= new Instances(testData,0),Cluster2 = new Instances(testData,0);

        for (int i : clusterSize) {i=0;}
        for (int i = cA.length - 1; i >= 0; i--) {
                clusterSize[(int) cA[i]]++;
            if(cA[i]==0){Cluster0.add(testData.get(i));}
            if(cA[i]==1){Cluster1.add(testData.get(i));}
            if(cA[i]==2){Cluster2.add(testData.get(i));}
            scoreList.add(testData.get(i).attribute(7));
        }
        for(int j=0;j<AnzahlCluster;j++)
        {
            System.out.println("cluster " + j + " has length " + clusterSize[j]);
        }

        Plot2D offScreenPlot = new Plot2D();
        offScreenPlot.setSize(500,500);

        PlotData2D master = new PlotData2D(Cluster0);
        master.setPlotName("Cluster 0 Score vs. Hits");

        //BufferedImage osi = weka.gui.beans.WekaOffscreenChartRenderer(500,500,Cluster0,"pim","ppg",[])

        master.m_displayAllPoints = true;
        try {
            offScreenPlot.setMasterPlot(master);
            offScreenPlot.addPlot(new PlotData2D(Cluster1));
            offScreenPlot.addPlot(new PlotData2D(Cluster2));

        } catch (Exception e) {
            e.printStackTrace();
        }
        BufferedImage osi = new BufferedImage(500, 500,
                BufferedImage.TYPE_INT_RGB);
        // render
        // plotting axes and color
        offScreenPlot.setXindex(5);
        offScreenPlot.setYindex(6);
        offScreenPlot.setCindex(2);


        java.awt.Graphics g = osi.getGraphics();
        offScreenPlot.paintComponent(g);


        final javax.swing.JFrame jf = new javax.swing.JFrame("Test");
        jf.setSize(600, 600);
        jf.getContentPane().setLayout(new java.awt.BorderLayout());
        // write the file
      jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
                System.exit(0);
            }
        });

//        jf.getContentPane().add(offScreenPlot, java.awt.BorderLayout.CENTER);
//        jf.setVisible(true);

        try {
            ImageIO.write(osi, "png", new File("/home/kos/Code/kos/dataMining/dataSetsNHL/snapshot.png"));
        } catch (IOException e) {
            e.printStackTrace();
        }


//        int xAx = offScreenPlot.getIndexOfAttribute(masterInstances, xAxis);
//        int yAx = getIndexOfAttribute(masterInstances, yAxis);
//        if (xAx < 0) {
//            xAx = 0;
//        }
//        if (yAx < 0) {
//            yAx = 0;
//        }
//        // plotting axes and color
        offScreenPlot.setXindex(5);
        offScreenPlot.setYindex(6);
        DecimalFormat df = new DecimalFormat("#.0");

        int tableRowZaehler=0;
        try {
            HtmlCanvas html = new HtmlCanvas();
            html
                    .head()
                    .title().content("data evaluation of NHL data")
                    .macros().stylesheet("htdocs/style-01.css")
                    .render(JQueryLibrary.core("1.4.3"))
                    .render(JQueryLibrary.ui("1.8.6"))
                    .render(JQueryLibrary.baseTheme("1.8"))
                   // .render(JQueryLibrary.mobile("1.0a3"))
                    ._head()
                    .body()
                    .h1().content("We got the following results from the cluster analysis \n")
                    .a().content("We have "+AnzahlCluster+" clusters!")
                    //.h1().content(eval.clusterResultsToString())
                    //._textarea()
                    .table(class_("cluster-summary-table"))
                    .img(src("/home/kos/Code/kos/dataMining/dataSetsNHL/snapshot.png").alt("scatter plot"))
                    .a().content("The following table shows mean values and standard deviations of the centroids")
                    .tr()
                    .th().content("Attribute")
                    .th().content("full data")
                    .th().content("Cluster 0")
                    .th().content("Cluster 1")
                    ._tr()
                    ._table()
                    ._body();
                    for(int i=0;i<testData.numAttributes()-1;i++) {
                        html
                                .body()

                                .table(class_("cluster-summary-table"))
                                .tr()
                                .td().content(testData.attribute(++tableRowZaehler).name())
                                .td().content(String.valueOf(df.format(testData.attributeStats(tableRowZaehler).numericStats.mean)+"\t") + " +- " + String.valueOf(df.format(testData.attributeStats(tableRowZaehler).numericStats.stdDev)))
                                .td().content(String.valueOf(df.format(Cluster0.attributeStats(tableRowZaehler).numericStats.mean)+"\t") + " +- " + String.valueOf(df.format(Cluster0.attributeStats(tableRowZaehler).numericStats.stdDev)))
                                .td().content(String.valueOf(df.format(Cluster1.attributeStats(tableRowZaehler).numericStats.mean)+"\t") + " +- " + String.valueOf(df.format(Cluster1.attributeStats(tableRowZaehler).numericStats.stdDev)))
                                ._tr()
                                ._table()
                                ._body();
                    }
/*                    .tr()
                    .td().content(testData.attribute(++tableRowZaehler).name())
                    .td().content(String.valueOf(df.format(testData.attributeStats(tableRowZaehler).numericStats.mean))+" +- "+String.valueOf(df.format(testData.attributeStats(tableRowZaehler).numericStats.stdDev)))
                    .td().content(String.valueOf(df.format(Cluster0.attributeStats(tableRowZaehler).numericStats.mean))+" +- "+String.valueOf(df.format(Cluster0.attributeStats(tableRowZaehler).numericStats.stdDev)))
                    .td().content(String.valueOf(df.format(Cluster1.attributeStats(tableRowZaehler).numericStats.mean))+" +- "+String.valueOf(df.format(Cluster1.attributeStats(tableRowZaehler).numericStats.stdDev)))
                    ._tr()
                    .tr()
                    .td().content(testData.attribute(++tableRowZaehler).name())
                    .td().content(String.valueOf(df.format(testData.attributeStats(tableRowZaehler).numericStats.mean))+" +- "+String.valueOf(df.format(testData.attributeStats(tableRowZaehler).numericStats.stdDev)))
                    .td().content(String.valueOf(df.format(Cluster0.attributeStats(tableRowZaehler).numericStats.mean))+" +- "+String.valueOf(df.format(Cluster0.attributeStats(tableRowZaehler).numericStats.stdDev)))
                    .td().content(String.valueOf(df.format(Cluster1.attributeStats(tableRowZaehler).numericStats.mean))+" +- "+String.valueOf(df.format(Cluster1.attributeStats(tableRowZaehler).numericStats.stdDev)))
                    ._tr()

                    ._table()
                   ._body();*/
               final String rendered = html.toHtml();
               final File output;
                output = new File("/home/kos/Code/kos/dataMining/output.html");
            Files.write(output.toPath(), rendered.getBytes("UTF-8"), StandardOpenOption.CREATE);
        } catch (IOException e) {
            e.printStackTrace();
        }


        String TST=j2html.TagCreator.html().with(
                j2html.TagCreator.head().with(
                        title("Title"),
                        j2html.TagCreator.link().withRel("stylesheet").withHref("/css/main.css")
                ),
                j2html.TagCreator.body().with(
                        j2html.TagCreator.main().with(
                                j2html.TagCreator.h1("Heading!"),
                                j2html.TagCreator.table().with()
                        ),
                        j2html.TagCreator.a("next")
                )
        ).render();
        final String rendered2 = j2html.TagCreator.html().toString();
        final File output2;
        System.out.println(TST);
        output2 = new File("/home/kos/Code/kos/dataMining/output2.html");
        //Files.write(output2.toPath(),rendered2.getBytes("UTF-8"), StandardOpenOption.CREATE);



        com.webfirmframework.wffweb.tag.html.Html html = new com.webfirmframework.wffweb.tag.html.Html(null) {
            com.webfirmframework.wffweb.tag.html.Body body = new com.webfirmframework.wffweb.tag.html.Body(this, new

                    com.webfirmframework.wffweb.tag.html.attribute.global.Style

                    (new com.webfirmframework.wffweb.css.BackgroundColor(CssColorName.LIGHT_CYAN.getColorName()))) {
                com.webfirmframework.wffweb.tag.html.tables.Table table = new com.webfirmframework.wffweb.tag.html.tables.Table(this) {

                    {


                        com.webfirmframework.wffweb.tag.html.tables.Tr

                                tr = new

                                com.webfirmframework.wffweb.tag.html.tables.Tr

                                        (this) {


                                    com.webfirmframework.wffweb.tag.html.tables.Td

                                            td1 = new

                                    com.webfirmframework.wffweb.tag.html.tables.Td

                                            (this) {


                                        com.webfirmframework.wffweb.tag.htmlwff.Blank

                                                cellContent = new

                                        com.webfirmframework.wffweb.tag.htmlwff.Blank

                                        (this,
                                        "column1 value");
                            };


                                    com.webfirmframework.wffweb.tag.html.tables.Td

                                            td2 = new

                                    com.webfirmframework.wffweb.tag.html.tables.Td

                                            (this) {


                                        com.webfirmframework.wffweb.tag.htmlwff.Blank

                                                cellContent = new

                                        com.webfirmframework.wffweb.tag.htmlwff.Blank

                                        (this,
                                        "column2 value");
                            };


                                    com.webfirmframework.wffweb.tag.html.tables.Td

                                            td3 = new

                                    com.webfirmframework.wffweb.tag.html.tables.Td

                                            (this) {


                                        com.webfirmframework.wffweb.tag.htmlwff.Blank

                                                cellContent = new

                                                com.webfirmframework.wffweb.tag.htmlwff.Blank

                                                (this,
                                        "column3 value");
                            };
                        };
                    }

                };
            };
        };

        html.setPrependDocType(true);
        System.out.println(html.toHtmlString());


//        testInstance.readNHLSumDataBufferedReader();

        // Increment B
       // testInstance.doIt();
       // testInstance.getAccuracy();

        // Call toString() method
        System.out.println("finished demo routine of testing java class! ");
    }
}