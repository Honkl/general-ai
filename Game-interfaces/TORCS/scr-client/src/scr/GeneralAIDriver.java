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

    /* Gear Changing Constants*/
    final int[] gearUp = {5000, 6000, 6000, 6500, 7000, 0};
    final int[] gearDown = {0, 2500, 3000, 3000, 3500, 3500};

    private BufferedWriter writer;
    private BufferedReader reader;
    private Process p;
    private GearInterval[] intervals;

    private SensorModel lastSensor;
    private double lastDistanceRaced = 0;
    private double bestDistanceRaced = 0;
    private double score = 0;

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
        initIntervals();

        writer = new BufferedWriter(new OutputStreamWriter(System.out));
        reader = new BufferedReader(new InputStreamReader(System.in));

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
        Action act = null;
        try {
            double raced = sensors.getDistanceRaced();
            if (raced != Double.NaN && raced != Double.NEGATIVE_INFINITY && raced != Double.POSITIVE_INFINITY) {
                bestDistanceRaced = Math.max(raced, bestDistanceRaced);
            }
            JsonMessageObject jmo = new JsonMessageObject(sensors, evaluateReward(), bestDistanceRaced, score, false);
            String json = jmo.convertToJson() + "\n";

            writer.write(json);
            writer.flush();

            // Debug
            //for (int i = 0; i < 20; i++) {
            //    System.err.println(reader.readLine());
            //}
            String output = reader.readLine();
            if (output == null) {
                System.err.println("WRONG AI RESULT");
            }

            String[] parts = output.split(" ");
            if (parts.length != 3) {
                System.err.println("Incorrect number of results from AI. Expecting 3, got " + parts.length);
            }
            double[] values = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                values[i] = Double.parseDouble(parts[i]);
            }

            // AI results
            act = new Action();
            act.accelerate = values[0];
            act.brake = values[1];
            act.steering = getSteer(values[2]);

            // Non-AI results
            act.gear = getGear(sensors);
            act.clutch = 0;
            act.focus = 0;
            act.restartRace = false;

            lastSensor = sensors;
            if (lastSensor.getDamage() == 0) {
                double lapTime = lastSensor.getCurrentLapTime();
                double distance = lastSensor.getDistanceRaced();
                if (lapTime > 0) {
                    // Measure only by distance
                    score = distance;
                    
                    /** // Measure by distance and laptime
                    if (lapTime < 1) {
                        score = distance;
                    } else {
                        score = distance / lapTime;
                    }
                    /**/
                } else {
                    score = 0;
                }
            }

            return act;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return act;
    }

    private double evaluateReward() {
        if (lastSensor == null) {
            return 0;
        }
        double speed = lastSensor.getSpeed();
        double angle = lastSensor.getAngleToTrackAxis();
        double pos = lastSensor.getTrackPosition();
        double reward = (speed * Math.cos(angle)) - (speed * Math.sin(angle)) - (speed * Math.abs(pos));
        // double reward = speed - (speed * Math.abs(lastSensor.getTrackPosition())); // Old reward
        return reward;
    }

    /**
     * Transforms AI's output in interval [0, 1] to proper interval for the
     * steer.
     *
     * @param outputFromAi Output of the AI.
     */
    private double getSteer(double outputFromAi) {
        // Steer must be in [-1, 1] interval
        double steer = (2 * outputFromAi) - 1;
        return steer;
    }

    /**
     * Reused method from SimpleDriver.java.
     *
     * @param sensors Sensors from the server.
     * @return Gear to use.
     */
    private int getGear(SensorModel sensors) {

        int gear = sensors.getGear();
        double rpm = sensors.getRPM();

        // if gear is 0 (N) or -1 (R) just return 1 
        if (gear < 1) {
            return 1;
        }
        // check if the RPM value of car is greater than the one suggested 
        // to shift up the gear from the current one     
        if (gear < 6 && rpm >= gearUp[gear - 1]) {
            return gear + 1;
        } else // check if the RPM value of car is lower than the one suggested 
        // to shift down the gear from the current one
        {
            if (gear > 1 && rpm <= gearDown[gear - 1]) {
                return gear - 1;
            } else // otherwhise keep current gear
            {
                return gear;
            }
        }
    }

    @Override
    public void reset() {
        System.out.println("Restarting the race!");
    }

    @Override
    public void shutdown() {
        JsonMessageObject jmo = new JsonMessageObject(lastSensor, evaluateReward(), bestDistanceRaced, score, true);
        try {
            writer.write(jmo.convertToJson() + "\n");
            writer.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
