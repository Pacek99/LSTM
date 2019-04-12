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
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Level;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacv.cvkernels;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVMultiSequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Patrik
 */
public class LSTMNeuronka {
    
    private static final Logger log = LoggerFactory.getLogger(UCISequenceClassificationExample.class);

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("src/main/resources/uci/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");
    
    private static int filesCount;
    
    //variables from Neuronka project
    private static Map<Double, Double> map;
    private static String sensor = "AK09918C";
    private static String processedDataFile = "processedData.csv";
    private static List<double[]> oneActivity;
    private static String csvSplitBy = "\t";

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
        
        filesCount = 0;
        
        // TODO code application logic here
        Map<String, String> subory = new HashMap<String,String>();
        
        // prieèinok B. Taylorová
        subory.put("src/main/resources/B. Taylorová/indora-1549482603748.csv", "MPU6500 Acceleration Sensor");        
        subory.put("src/main/resources/B. Taylorová/indora-1549482689471.csv", "MPU6500 Acceleration Sensor");        
        subory.put("src/main/resources/B. Taylorová/indora-1549482748787.csv", "MPU6500 Acceleration Sensor");        
        subory.put("src/main/resources/B. Taylorová/indora-1549482827151.csv", "MPU6500 Acceleration Sensor");        
        subory.put("src/main/resources/B. Taylorová/indora-1549482928447.csv", "MPU6500 Acceleration Sensor");        
        subory.put("src/main/resources/B. Taylorová/indora-1549482960030.csv", "MPU6500 Acceleration Sensor");
                      
        //subory.put("src/main/resources/B. Taylorová/indora-1552906231896.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1552906271137.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1552906425702.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1552906472467.csv", "MPU6500 Acceleration Sensor");
               
        //subory.put("src/main/resources/B. Taylorová/indora-1552906553847.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1553278125310.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1553278129057.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1553278160954.csv", "MPU6500 Acceleration Sensor");  
        
        // prieèinok M. Sochuliak
        subory.put("src/main/resources/M. Sochuliak/indora-1549012172677.csv", "ACCELEROMETER");        
        subory.put("src/main/resources/M. Sochuliak/indora-1549021777198.csv", "ACCELEROMETER");        
        subory.put("src/main/resources/M. Sochuliak/indora-1549022025135.csv", "ACCELEROMETER");        
        subory.put("src/main/resources/M. Sochuliak/indora-1549022068213.csv", "ACCELEROMETER");
        
        // prieèinok P. Kendra
        subory.put("src/main/resources/P. Kendra/indora-1549541475108.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541522778.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541559653.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541572516.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541582979.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541633702.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541647313.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541673057.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541682687.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541695967.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541708344.csv", sensor);
        
        // prieèinok P. Rojek
        subory.put("src/main/resources/P. Rojek/indora-1540484172540.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1540484443308.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1540484680716.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1540546792759.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1540554936805.csv", sensor);
        subory.put("src/main/resources/P. Rojek/indora-1554699869876.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554699898743.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554699925682.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554699994623.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554710761298.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554728381607.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554728410434.csv", sensor);
        
        
        // prieèinok Š. Rojek
        subory.put("src/main/resources/Š. Rojek/indora-1540362934669.csv", sensor);        
        subory.put("src/main/resources/Š. Rojek/indora-1540363171233.csv", sensor);        
        subory.put("src/main/resources/Š. Rojek/indora-1540363247042.csv", sensor);        
        subory.put("src/main/resources/Š. Rojek/indora-1540363314900.csv", sensor);        
        subory.put("src/main/resources/Š. Rojek/indora-1540363406576.csv", sensor);
        
        // prieèinok sk.upjs.indora.sensorsrecorder         
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541475108.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541522778.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541552672.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541559653.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541572516.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541582979.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541633702.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541647313.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541673057.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541682687.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541695967.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541708344.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553014481636.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553014543983.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553014570035.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553019600096.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553019658511.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553019699799.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589053903.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589171764.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589248068.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589338226.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589368041.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589528999.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609125909.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609140518.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609169349.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609197288.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609224970.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609253485.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553610218773.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553610244950.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553610273270.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553610302262.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553610326669.csv", sensor); 
        
        // vygenerovanie datasetu 
        for (Map.Entry<String, String> entry : subory.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            System.out.println("Subor: " + key + " a akcelerometer: " + value);
            processData(value,key);
        }    
        
        //LSTM neuronka odtial dalej
        // ----- Load the training data -----
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(0, csvSplitBy);
        //SequenceRecordReader trainFeatures = new CSVMultiSequenceRecordReader(csvSplitBy, CSVMultiSequenceRecordReader.Mode.EQUAL_LENGTH);
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        try {
            trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, filesCount-1));
            trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, filesCount-1));
        } catch (IOException ex) {
            java.util.logging.Logger.getLogger(LSTMNeuronka.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InterruptedException ex) {
            java.util.logging.Logger.getLogger(LSTMNeuronka.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        int miniBatchSize = 10;
        int numLabelClasses = 6;        
        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        trainData.setPreProcessor(normalizer);

        /*
        // ----- Load the test data -----
        //Same process as for the training data.
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
         
        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        
        testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data
        */

        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.005))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(3).nOut(10).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));   //Print the score (loss function value) every 20 iterations


        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 40;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);
/*
            //Evaluate on the test set:
            Evaluation evaluation = net.evaluate(testData);
            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

            testData.reset();
*/
            trainData.reset();
        }

        log.info("----- Example Complete -----");
    }
    
    public static void processData(String sensor, String csvFile){
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
                            compute(oneActivity);                             
                            
                            oneActivity = new ArrayList<>();
                            break;
                        }
                    } 
                    //ak sme došli na koniec suboru a posledna aktivita vo file tak ju spracujeme
                    if (oneActivity.size() != 0) {
                        System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                        compute(oneActivity); 
                        
                        oneActivity = new ArrayList<>();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private static void compute(List<double[]> oneActivity) {
        String features = "";
        for (double[] entry : oneActivity) {
            features = features + entry[1] + csvSplitBy + entry[2] + csvSplitBy + entry[3] + "\n";
        }
        
        //Write output in a format we can read, in the appropriate locations
        File outPathFeatures = new File(featuresDirTrain, filesCount + ".csv");
        File outPathLabels = new File(labelsDirTrain, filesCount + ".csv");
       
        try {
            FileUtils.writeStringToFile(outPathFeatures, features);
            FileUtils.writeStringToFile(outPathLabels, String.valueOf((int)oneActivity.get(0)[4]));
        } catch (IOException ex) {
            java.util.logging.Logger.getLogger(LSTMNeuronka.class.getName()).log(Level.SEVERE, null, ex);
        }
        filesCount++;
    }  
}
