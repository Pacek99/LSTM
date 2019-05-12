/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.lstm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Patrik
 */
public class ExternalDatasetVersion {
    
    private static final Logger log = LoggerFactory.getLogger(UCISequenceClassificationExample.class);

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("src/main/resources/uciExternalDataset/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");
    
    private static int trainCount;
    private static int testCount;
    
    //variables from Neuronka project
    private static Map<Double, Double> map;
    private static String sensor = "ST_LIS3DHTR";
    private static String processedDataFile = "processedData.csv";
    private static List<double[]> oneActivity;
    private static String csvSplitBy = "\t";
    
    //pomocne pole na zistenie poctu jednotlivych aktivit v externom datasete
    private static int[] poctyAktivit;
    private static int[] poctyNaTraining;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        //Create directories
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();
        
        //ak už mame dataset spracovany
        trainCount = 634;
        testCount = 264;
        
        //trainCount = 0;
        //testCount = 0;
        
        //na pevno nastavene hodnoty pre externy dataset 70% train a 30% test
        poctyAktivit = new int[5];
        poctyNaTraining = new int[5];
        poctyNaTraining[0] = 47;
        poctyNaTraining[1] = 48;
        poctyNaTraining[2] = 243;
        poctyNaTraining[3] = 243;
        poctyNaTraining[4] = 48;
        
        
        // TODO code application logic here
        Map<String, String> subory = new HashMap<String,String>();
        Map<String, String> suboryNaTest = new HashMap<String,String>();
        
        //nacitanie suborov na vytah hore                
        subory.put("src/main/resources/Vytah/indora-1540546792759.csv", sensor);        
        subory.put("src/main/resources/Vytah/indora-1549541647313.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554699869876.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554699925682.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554728410434.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556082478032.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556082533389.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556110357233.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556110955057.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111007555.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111077350.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111125321.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111174552.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111227247.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111281572.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111329156.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111383811.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111433500.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111481240.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111530481.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111618750.csv", sensor);
        
        //nacitanie suborov na vytah dole
        subory.put("src/main/resources/Vytah/indora-1540554936805.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1549541522778.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554699898743.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554699994623.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554710761298.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554728381607.csv", sensor);        
        subory.put("src/main/resources/Vytah/indora-1556082506828.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556082560909.csv", sensor);      
        subory.put("src/main/resources/Vytah/indora-1556110062122.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556110929890.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556110980942.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111051918.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111098861.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111149569.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111200762.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111304040.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111352315.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111407332.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111459842.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111504747.csv", sensor);
        suboryNaTest.put("src/main/resources/Vytah/indora-1556111665614.csv", sensor);
         
        /*
        // vygenerovanie treningoveho datasetu 
        for (Map.Entry<String, String> entry : subory.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            System.out.println("Subor: " + key + " a akcelerometer: " + value);
            processData(value,key,true);
        }    
        
        // vygenerovanie testovcieho datasetu 
        for (Map.Entry<String, String> entry : suboryNaTest.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            System.out.println("Subor: " + key + " a akcelerometer: " + value);
            processData(value,key,false);
        }*/
        
        /*
        //vygenerovanie datasetov z externeho datasetu
        processDataExternalDataset("src/main/resources/Phones_accelerometer.csv");
        for (int i = 0; i < poctyAktivit.length; i++) {
            System.out.println("pocet zaznamov aktivity " + i + "  :" + poctyAktivit[i]);
        }
        */
        //LSTM neuronka odtial dalej
        // ----- Load the training data -----
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(0, csvSplitBy);
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        try {
            trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, trainCount-1));
            trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, trainCount-1));
        } catch (IOException ex) {
            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InterruptedException ex) {
            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        int miniBatchSize = 5;
        //int numLabelClasses = 6;
        int numLabelClasses = 5;   //externy dataset     
        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
                     
        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        //trainData.setPreProcessor(normalizer);
        trainData.setPreProcessor(new CompositeDataSetPreProcessor(normalizer, new LabelLastTimeStepPreProcessor()));

        
        // ----- Load the test data -----
        //Same process as for the training data.
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader(0, csvSplitBy);
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        try {
            testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, testCount-1));        
            testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, testCount-1));
        } catch (IOException ex) {
            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InterruptedException ex) {
            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        } 
        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        
        //testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data
        testData.setPreProcessor(new CompositeDataSetPreProcessor(normalizer, new LabelLastTimeStepPreProcessor()));
        
        /*
        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.005))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new LSTM.Builder().activation(Activation.TANH).updater(new Adam(1e-3)).nIn(3).nOut(10).build())
                .layer(1, new LSTM.Builder().activation(Activation.TANH).nIn(10).nOut(10).build())
                .layer(2, new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build())
                .layer(3, new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                .layer(4, new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH).build())
                .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build())
                .build();
        */
        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.005))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .seed(123)
                .graphBuilder()
                .addInputs("input")
                .addLayer("L1", new LSTM.Builder().nIn(3).nOut(5).activation(Activation.TANH).build(),"input")
                .addLayer("L2", new LSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build(),"L1")
                .addLayer("globalPoolMax", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).dropOut(0.5).build(), "L2")                
                .addLayer("globalPoolAvg", new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).dropOut(0.5).build(), "globalPoolMax")
                .addLayer("D1", new DenseLayer.Builder().nIn(5).nOut(5).activation(Activation.TANH).build(), "globalPoolAvg")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(5).nOut(numLabelClasses).build(), "D1")
                .setOutputs("output")
                .build();

        ComputationGraph net = new ComputationGraph(configuration);
        
        //MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));   //Print the score (loss function value) every 20 iterations

        
        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 10;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);

            //Evaluate on the test set:
            //Evaluation evaluation = net.evaluate(testData);            
            //log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));
                       
            Evaluation evaluation = net.evaluate(testData); 
            log.info(evaluation.stats());
        
            testData.reset();
            trainData.reset();
        }
        Evaluation evaluation = net.evaluate(testData); 
        log.info(evaluation.stats());

        log.info("----- Example Complete -----");
    }
    
    public static void processData(String sensor, String csvFile, boolean forTraining){
        String line = "";
	String cvsSplitBy = "\t";
        String currentActivity;
        String activityCode = "";
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            while ((line = br.readLine()) != null) {
                String[] zaznam = line.split(cvsSplitBy);
                currentActivity = zaznam[1];
                
                if (zaznam[2].equals(sensor)) {                     
                    if (currentActivity.equals("standing")) {
                        activityCode = "0";
                    } 
                    if (currentActivity.equals("walking")) {
                        activityCode = "1";
                    }
                    if (currentActivity.equals("walkingUpstairs")) {
                        activityCode = "2";
                    }
                    if (currentActivity.equals("walkingDownstairs")) {
                        activityCode = "3";
                    }
                    if (currentActivity.equals("elevatorUp")) {
                        activityCode = "4";
                    }
                    if (currentActivity.equals("elevatorDown")) {
                        activityCode = "5";
                    }
                    
                   
                    oneActivity = new ArrayList<>();
                    //add values for time, xAxis, yAxis, zAxis, activityCode
                    oneActivity.add(new double[]{Double.valueOf(zaznam[0]), Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                                    Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                    while ((line = br.readLine()) != null) {
                        zaznam = line.split(cvsSplitBy);
                        if (zaznam[1].equals(currentActivity)) {
                            if (zaznam[2].equals(sensor)) {  
                                //add values for time, xAxis, yAxis, zAxis, activityCode
                                oneActivity.add(new double[]{Double.valueOf(zaznam[0]), Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                                    Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                            }                            
                        } else {
                            //tu spracovat data aktivity a dat do suboru
                            System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                            compute(oneActivity, forTraining);                             
                            
                            oneActivity = new ArrayList<>();
                            break;
                        }
                    } 
                    //ak sme došli na koniec suboru a posledna aktivita vo file tak ju spracujeme
                    if (oneActivity.size() != 0) {
                        System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                        compute(oneActivity, forTraining); 
                        
                        oneActivity = new ArrayList<>();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private static void compute(List<double[]> oneActivity, boolean forTraining) {
        String features = "";
        for (double[] entry : oneActivity) {
            features = features + entry[1] + csvSplitBy + entry[2] + csvSplitBy + entry[3] + "\n";
        }
        
        //Write output in a format we can read, in the appropriate locations
        File outPathFeatures;
        File outPathLabels;
        if (forTraining) {
            outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
            outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
        } else {
            outPathFeatures = new File(featuresDirTest, testCount + ".csv");
            outPathLabels = new File(labelsDirTest, testCount + ".csv");
        }
        
       
        try {
            FileUtils.writeStringToFile(outPathFeatures, features);
            FileUtils.writeStringToFile(outPathLabels, String.valueOf((int)oneActivity.get(0)[4]));
        } catch (IOException ex) {
            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        if (forTraining) {
            trainCount++;
        } else {
            testCount++;
        }        
    }  
    
    public static void processDataExternalDataset(String csvFile){
        String line = "";
        String cvsSplitBy = ",";
        String currentActivity;
        String activityCode = "";
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            br.readLine();//preskoc prvy riadok s nazvami stlpcov
            while ((line = br.readLine()) != null) {
                String[] zaznam = line.split(cvsSplitBy);
                
                currentActivity = zaznam[9];
                
                switch(currentActivity){
                    case "stand":
                        activityCode = "0";
                        break;
                    case "walk":
                        activityCode = "1";
                        break;
                    case "stairsup":
                        activityCode = "2";
                        break;
                    case "stairsdown":
                        activityCode = "3";
                        break;
                    case "sit":
                        activityCode = "4";
                        break;
                    default:
                        //aktivita ktoru neuvazujeme
                        activityCode = "10";
                }
                
                if (activityCode.equals("10")) {
                    continue;
                }
                   
                oneActivity = new ArrayList<>();
                //add values for xAxis, yAxis, zAxis, activityCode
                oneActivity.add(new double[]{Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                                Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                while ((line = br.readLine()) != null) {
                    zaznam = line.split(cvsSplitBy);
                    if (zaznam[9].equals(currentActivity)) {                          
                        //add values for xAxis, yAxis, zAxis, activityCode
                        oneActivity.add(new double[]{Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                                Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});                          
                    } else {
                        //tu spracovat data aktivity a dat do suboru
                        System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                        if (poctyAktivit[Integer.parseInt(activityCode)] <= poctyNaTraining[Integer.parseInt(activityCode)]) {
                            poctyAktivit[Integer.parseInt(activityCode)]++;
                            computeExternalDataset(oneActivity, true); 
                        } else {
                            computeExternalDataset(oneActivity, false); 
                        }                                                
                        
                        oneActivity = new ArrayList<>();
                        break;
                    }
                } 
                //ak sme došli na koniec suboru a posledna aktivita vo file tak ju spracujeme
                if (oneActivity.size() != 0) {
                    System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                    computeExternalDataset(oneActivity, true); //true len preto lebo mame len 1 file takze na jednej aktivite uz nezalezi ci train/test
                    poctyAktivit[Integer.parseInt(activityCode)]++;
                    oneActivity = new ArrayList<>();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } 
    }
    
    private static void computeExternalDataset(List<double[]> oneActivity, boolean forTraining) {
        String features = "";
        for (double[] entry : oneActivity) {
            features = features + entry[0] + csvSplitBy + entry[1] + csvSplitBy + entry[2] + "\n";
        }
        
        //Write output in a format we can read, in the appropriate locations
        File outPathFeatures;
        File outPathLabels;
        
        if (forTraining) {
            outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
            outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
        } else {
            outPathFeatures = new File(featuresDirTest, testCount + ".csv");
            outPathLabels = new File(labelsDirTest, testCount + ".csv");
        }
               
        try {
            FileUtils.writeStringToFile(outPathFeatures, features);
            FileUtils.writeStringToFile(outPathLabels, String.valueOf((int)oneActivity.get(0)[3]));
        } catch (IOException ex) {
            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        if (forTraining) {
            trainCount++;
        } else {
            testCount++;
        }                 
    }  
}
