/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package scr;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

/**
 *
 * @author Jan Kluj
 */
public class GeneralAIDriver extends Controller {

    private BufferedWriter writer;
    private BufferedReader reader;
    private Process p;
    private GearInterval[] intervals;
    private SimpleDriver sd;
    
    //Config file for AI (relative path to master "general-ai/Game-interfaces" directory
    private final String configFile = "Game-interfaces\\TORCS\\TORCS_config.txt"; 
    
    private SensorModel lastSensor;

    /**
     * Represents an interval to discretise values for gear integer.
     */
    private class GearInterval {

        public double lowerBound;
        public double upperBound;

        public GearInterval(double lowerbound, double upperBound) {
            this.lowerBound = lowerbound;
            this.upperBound = upperBound;
        }
    }

    @Override
    public float[] initAngles() {

        sd = new SimpleDriver();
        initIntervals();

        try {
            ProcessBuilder pb = new ProcessBuilder(new String[]{Client.PythonExe, Client.PythonScriptFile, configFile});
            pb.redirectErrorStream(true);

            p = pb.start();
            writer = new BufferedWriter(new OutputStreamWriter(p.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(p.getInputStream()));

            Thread.sleep(100);
        } catch (IOException ex) {
            System.out.println("Exception while runtime.exec");
            ex.printStackTrace();
        } catch (InterruptedException ex) {
            ex.printStackTrace();
        }

        float[] angles = new float[19];

        /* set angles as {-90,-75,-60,-45,-30,-20,-15,-10,-5,0,5,10,15,20,30,45,60,75,90} */
        for (int i = 0; i < 5; i++) {
            angles[i] = -90 + i * 15;
            angles[18 - i] = 90 - i * 15;
        }

        for (int i = 5; i < 9; i++) {
            angles[i] = -20 + (i - 5) * 5;
            angles[18 - i] = 20 - (i - 5) * 5;
        }
        angles[9] = 0;
        return angles;
    }

    /**
     * Initializes intervals for discretization of gear.
     */
    private void initIntervals() {
        int numberOfIntervals = 8;
        double bias = 1.0 / 8.0;

        intervals = new GearInterval[numberOfIntervals];
        for (int i = 0; i < intervals.length; i++) {
            intervals[i] = new GearInterval(bias * i, bias * (i + 1));
        }
    }

    @Override
    public Action control(SensorModel sensors) {
        try {
            JsonMessageObject jmo = new JsonMessageObject(sensors);
            String json = jmo.convertToJson() + "\n";

            writer.write(json);
            writer.flush();

            String[] output = reader.readLine().split(" ");
            double[] values = new double[output.length];
            for (int i = 0; i < output.length; i++) {
                values[i] = Double.parseDouble(output[i]);
            }

            Action act = new Action();
            act.accelerate = values[0];
            act.brake = values[1];
            act.clutch = values[2];
            act.focus = getFocus(values[3]);
            act.gear = getGear(values[4]);
            act.steering = getSteer(values[5]);

            act.restartRace = false;
            lastSensor = sensors;
            
            return act;
        } catch (IOException e) {
            e.printStackTrace();
        }

        //System.out.println("Distance from start line: " + sensors.getDistanceFromStartLine());
        System.out.println("Distance raced: " + sensors.getDistanceRaced());
        return sd.control(sensors);
        /**
         * Action a = new Action(); a.gear = 1; a.accelerate = 0.2; return a;
        /*
         */
    }

    /**
     * Transforms AI's output in interval [0, 1] to proper interval for the steer.
     * @param outputFromAi Output of the AI.
     */
    private double getSteer(double outputFromAi) {
        // Steer must be in [-1, 1] interval
        return (2 * outputFromAi) - 1;
    }

    /**
     * Transforms AI's output in interval [0, 1] to proper gear integer.
     * @param outputFromAi Ouput of the AI.
     */
    private int getGear(double outputFromAi) {
        // Gear is in {-1, 0, ..., 6}
        for (int i = 0; i < intervals.length; i++) {
            if (outputFromAi >= intervals[i].lowerBound && outputFromAi <= intervals[i].upperBound) {
                return i - 1;
            }
        }
        return 0;
    }

    /**
     * Transforms AI's output in interval [0, 1] to proper focus integer.
     * @param outputFromAi Ouput of the AI.
     */
    private int getFocus(double outputFromAi) {
        // Focus must be in interval [-90, 90]
        return (int)((outputFromAi * 180) - 90);
    }

    @Override
    public void reset() {
        System.out.println("Restarting the race!");
    }

    @Override
    public void shutdown() {
        System.out.println("RACED DISTANCE: " + lastSensor.getDistanceRaced());
        try {
            writer.write("END");
            writer.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        System.out.println("Bye bye!");
    }

}
