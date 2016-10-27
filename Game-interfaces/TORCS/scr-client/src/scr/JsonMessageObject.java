package scr;

import com.google.gson.*;

/**
 *
 * @author Jan Kluj
 */
public class JsonMessageObject {


    //SensorModel sensors;
    
    // Special "property" (must have this name)
    int[] possible_moves;


    /**
     * Initializes a new instance of JsonMessageObject. Represents the current
     * state of the game.
     */
    public JsonMessageObject(SensorModel sensors) {
        //this.sensors = sensors;
        
        possible_moves = new int[1];
        possible_moves[0] = 1;
        
    }

    public int[] encodeMoves() {
        return null;
    }

    public void decodeMove(int inputForMario) {
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
