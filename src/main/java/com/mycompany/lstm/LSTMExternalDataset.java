package com.mycompany.lstm;

import com.mycompany.lstm.LabelLastTimeStepPreProcessor;
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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LSTMExternalDataset {

    private static final Logger log = LoggerFactory.getLogger(LSTMExternalDataset.class);

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("src/main/resources/uciLSTMExternalDataset/");
    //private static File baseDir = new File("C:/Temp/LSTM/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    private static int trainCount;
    private static int testCount;

    private static List<double[]> oneActivity;
    private static String csvSplitBy = "\t";

    private static int[] poctyAktivit;
    private static int[] numerOfSamplesForTraining;
    
    private static String sensor = "ST_LIS3DHTR";

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

        //if we already have dataset computed and commented lines 83-84 and 148-159
        trainCount = 668;
        testCount = 274;

        //trainCount = 0;
        //testCount = 0;

        //set value for external dataset 70% train a 30% test
        poctyAktivit = new int[7];
        numerOfSamplesForTraining = new int[7];
        numerOfSamplesForTraining[0] = 47;
        numerOfSamplesForTraining[1] = 48;
        numerOfSamplesForTraining[2] = 243;
        numerOfSamplesForTraining[3] = 243;
        numerOfSamplesForTraining[4] = 48;
        numerOfSamplesForTraining[5] = 16;
        numerOfSamplesForTraining[6] = 16;

        
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
        subory.put("src/main/resources/Vytah/indora-1556111329156.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111383811.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111433500.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111481240.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111530481.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111618750.csv", sensor);
        
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
        subory.put("src/main/resources/Vytah/indora-1556111304040.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111352315.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111407332.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111459842.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111504747.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111665614.csv", sensor);
         
        /*
        // vygenerovanie datasetu vytahov
        for (Map.Entry<String, String> entry : subory.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            System.out.println("Subor: " + key + " a akcelerometer: " + value);
            processDataVytah(key, value);
        }         

        //generate dataset from external dataset
        processDataExternalDataset("src/main/resources/Phones_accelerometer.csv");
        */      
        //LSTM neuronka odtial dalej
        // ----- Load the training data -----
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(0, csvSplitBy);
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        try {
            trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "\\%d.csv", 0, trainCount-1));
            trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "\\%d.csv", 0, trainCount-1));
        } catch (IOException ex) {
            ex.printStackTrace();
//            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InterruptedException ex) {
            ex.printStackTrace();
//            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        }

        int miniBatchSize = 5;
        int numLabelClasses = 7;
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
            ex.printStackTrace();
            //java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InterruptedException ex) {
            ex.printStackTrace();;
//            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        }
        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data
        testData.setPreProcessor(new CompositeDataSetPreProcessor(normalizer, new LabelLastTimeStepPreProcessor()));

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(0.005))
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
            .gradientNormalizationThreshold(0.5)
            .seed(123)
            .graphBuilder()
            .addInputs("input")
            .addLayer("L1", new LSTM.Builder().nIn(3).nOut(15).activation(Activation.TANH).build(),"input")
            .addLayer("L2", new LSTM.Builder().nIn(15).nOut(15).activation(Activation.TANH).build(),"L1")
            .addLayer("globalPoolMax", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).dropOut(0.5).build(), "L2")
            .addLayer("globalPoolAvg", new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).dropOut(0.5).build(), "L2")
            .addLayer("D1", new DenseLayer.Builder().nIn(30).nOut(15).activation(Activation.TANH).build(), "globalPoolMax", "globalPoolAvg")
            .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(15).nOut(numLabelClasses).build(), "D1")
            .setOutputs("output")
            .build();

        ComputationGraph net = new ComputationGraph(configuration);
        net.init();

        net.setListeners(new ScoreIterationListener(20));   //Print the score (loss function value) every 20 iterations


        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 120;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);

            Evaluation evaluation = net.evaluate(testData);
            log.info(evaluation.stats());

            testData.reset();
            trainData.reset();
        }
        Evaluation evaluation = net.evaluate(testData);
        log.info(evaluation.stats());

        log.info("----- Example Complete -----");
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
                        System.out.println("length of activity " + currentActivity + "  :" + oneActivity.size());
                        if (poctyAktivit[Integer.parseInt(activityCode)] <= numerOfSamplesForTraining[Integer.parseInt(activityCode)]) {
                            poctyAktivit[Integer.parseInt(activityCode)]++;
                            computeExternalDataset(oneActivity, true);
                        } else {
                            computeExternalDataset(oneActivity, false);
                        }

                        oneActivity = new ArrayList<>();
                        break;
                    }
                }
                //ak sme do?li na koniec suboru a posledna aktivita vo file tak ju spracujeme
                if (oneActivity.size() != 0) {
                    System.out.println("length of activity " + currentActivity + "  :" + oneActivity.size());
                    computeExternalDataset(oneActivity, true);
                    poctyAktivit[Integer.parseInt(activityCode)]++;
                    oneActivity = new ArrayList<>();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void processDataVytah(String csvFile, String senzor){
        String line = "";
        String cvsSplitBy = "\t";
        String currentActivity;
        String activityCode = "";
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            while ((line = br.readLine()) != null) {
                String[] zaznam = line.split(cvsSplitBy);
                currentActivity = zaznam[1];
                if (zaznam[2].equals(sensor)) {           
                    if (currentActivity.equals("elevatorUp")) {
                        activityCode = "5";
                    }
                    if (currentActivity.equals("elevatorDown")) {
                        activityCode = "6";
                    }                    
                   
                    oneActivity = new ArrayList<>();
                    //add values for xAxis, yAxis, zAxis, activityCode
                    oneActivity.add(new double[]{Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                    Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                    while ((line = br.readLine()) != null) {
                        zaznam = line.split(cvsSplitBy);
                        if (zaznam[1].equals(currentActivity)) {
                            if (zaznam[2].equals(sensor)) {
                                //add values for xAxis, yAxis, zAxis, activityCode
                                oneActivity.add(new double[]{Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                                Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                            }                            
                        } else {
                            System.out.println("length of activity " + currentActivity + "  :" + oneActivity.size());
                            if (poctyAktivit[Integer.parseInt(activityCode)] <= numerOfSamplesForTraining[Integer.parseInt(activityCode)]) {
                                poctyAktivit[Integer.parseInt(activityCode)]++;
                                computeExternalDataset(oneActivity, true);
                            } else {
                                computeExternalDataset(oneActivity, false);
                            }

                            oneActivity = new ArrayList<>();
                            break;
                        }
                    }   
                    //ak sme do?li na koniec suboru a posledna aktivita vo file tak ju spracujeme
                    if (oneActivity.size() != 0) {
                        System.out.println("length of activity " + currentActivity + "  :" + oneActivity.size());
                        if (poctyAktivit[Integer.parseInt(activityCode)] <= numerOfSamplesForTraining[Integer.parseInt(activityCode)]) {
                            poctyAktivit[Integer.parseInt(activityCode)]++;
                            computeExternalDataset(oneActivity, true);
                        } else {
                            computeExternalDataset(oneActivity, false);
                        }
                        oneActivity = new ArrayList<>();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private static void computeExternalDataset(List<double[]> oneActivity, boolean forTraining) {
        StringBuilder features = new StringBuilder();
        for (double[] entry : oneActivity) {
            features.append(entry[0]).append(csvSplitBy).append(entry[1]).append(csvSplitBy).append(entry[2]).append("\n");
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
            FileUtils.writeStringToFile(outPathFeatures, features.toString());
            FileUtils.writeStringToFile(outPathLabels, String.valueOf((int)oneActivity.get(0)[3]));
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        if (forTraining) {
            trainCount++;
        } else {
            testCount++;
        }
    }
}