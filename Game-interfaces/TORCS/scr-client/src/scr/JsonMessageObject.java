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


    /**
     * Initializes a new instance of JsonMessageObject. Represents the current
     * state of the game.
     * @param sensors Information about a game state.
     */
    public JsonMessageObject(SensorModel sensors) {
        List<Double> result = new ArrayList<>();
        
        result.add(sensors.getAngleToTrackAxis());
        result.add(sensors.getCurrentLapTime());
        result.add(sensors.getDamage());
        result.add(sensors.getDistanceFromStartLine());
        result.add(sensors.getDistanceRaced());
        result.add(sensors.getFuelLevel());
        result.add((double)sensors.getGear());
        result.add(sensors.getLastLapTime());
        result.add(sensors.getLateralSpeed());
        result.add(sensors.getRPM());
        result.add((double)sensors.getRacePosition());
        result.add(sensors.getSpeed());
        result.add(sensors.getTrackPosition());
        result.add(sensors.getZ());
        result.add(sensors.getZSpeed());
        
        for (double value : sensors.getFocusSensors()) {
            result.add(value);
        }
        
        for (double value : sensors.getOpponentSensors()) {
            result.add(value);
        }
        
        for (double value : sensors.getTrackEdgeSensors()) {
            result.add(value);
        }
        
        for (double value : sensors.getWheelSpinVelocity()) {
            result.add(value);
        }
        
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
