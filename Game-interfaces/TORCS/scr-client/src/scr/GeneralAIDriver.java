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
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Jan Kluj
 */
public class GeneralAIDriver extends Controller {

    private static BufferedWriter writer;
    private static BufferedReader reader;

    @Override
    public float[] initAngles() {

        // Start python process
        String pythonExePath = "C:\\Anaconda2\\envs\\py3k\\python.exe"; // TODO: use general relative path
        String pythonScriptPath = "..\\..\\..\\..\\Controller\\script.py";
        String classesDir = Client.GeneralAIDirectory;

        System.out.println(classesDir + "\\" + pythonScriptPath);

        Runtime rt = Runtime.getRuntime();
        Process p;
        try {
            p = rt.exec(new String[]{pythonExePath, classesDir + "\\" + pythonScriptPath});
            writer = new BufferedWriter(new OutputStreamWriter(p.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            //BufferedReader errReader = new BufferedReader(new InputStreamReader(p.getErrorStream()));
        } catch (IOException ex) {
            System.out.println("Exception while runtime.exec");
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

    @Override
    public Action control(SensorModel sensors) {
        try {
            JsonMessageObject jmo = new JsonMessageObject(sensors);
            String json = jmo.convertToJson() + "\n";
            System.out.println(json);
            writer.write(json);
            writer.flush();
            String output = reader.readLine();
            System.out.println(output);

        } catch (IOException e) {
            e.printStackTrace();
        }

        Action a = new Action();
        a.accelerate = 0.2;
        return a;
    }

    @Override
    public void reset() {
        System.out.println("Restarting the race!");
    }

    @Override
    public void shutdown() {
        System.out.println("Bye bye!");
    }

}
