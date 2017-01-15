package scr;

import com.google.gson.*;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Jan Kluj
 */
public class JsonMessageObject {

    Double[] state;
    int current_phase = 0;

    private transient double piRadians = 0.05483;

    /**
     * Initializes a new instance of JsonMessageObject. Represents the current
     * state of the game. All game sensor values are scaled for the AI.
     *
     * @param sensors Information about a game state.
     */
    public JsonMessageObject(SensorModel sensors) {
        List<Double> result = new ArrayList<>();

        result.add((sensors.getAngleToTrackAxis() + piRadians) / (2 * piRadians));
        result.add(Math.log(sensors.getCurrentLapTime() + 1));
        result.add(Math.log(sensors.getDamage() + 1));
        result.add(Math.log(sensors.getDistanceFromStartLine() + 1));
        result.add(Math.log(sensors.getDistanceRaced() + 1));
        result.add(Math.log(sensors.getFuelLevel() + 1));
        result.add((double) sensors.getGear() / 8.0);
        result.add(Math.log(sensors.getLastLapTime() + 1));
        result.add(Math.log(sensors.getLateralSpeed() + 1));
        result.add(Math.log(sensors.getRPM() + 1));
        result.add((double) sensors.getRacePosition());
        result.add(sensors.getSpeed());
        result.add(sensors.getTrackPosition());
        result.add(sensors.getZ());
        result.add(sensors.getZSpeed());

        for (double value : sensors.getFocusSensors()) {
            result.add(value / 200);
        }

        for (double value : sensors.getOpponentSensors()) {
            result.add(value / 200);
        }

        for (double value : sensors.getTrackEdgeSensors()) {
            result.add(value / 200);
        }

        // ABS should be done manually
        //for (double value : sensors.getWheelSpinVelocity()) { 
        //    result.add(value);
        //}
        // Debug print
        //for (int i = 0; i < result.size(); i++) {
        //    System.err.print(result.get(i) + " ");
        //}
        state = new Double[result.size()];
        result.toArray(state);
    }

    /**
     * Converts the current object to JSON representation (using Gson).
     *
     * @return the current object converted to JSON (as a String).
     */
    public String convertToJson() {
        Gson gson = new Gson();
        String json = gson.toJson(this);
        return json;
    }
}
