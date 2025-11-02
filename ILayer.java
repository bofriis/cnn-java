/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;

/**
 *
 * @author gola
 */
public interface ILayer {
    
    int get_out_sx();
    int get_out_sy();
    int get_out_depth();
    Vol get_out_act();
    String get_layer_type();
    
    Vol forward(Vol V, boolean is_training);
    void backward();
    ParamsAndGrads[] getParamsAndGrads(); 
    
    // Added methods to support internal accessors
    void set_in_act(Vol v);
    Vol get_in_act();
    
    public void save(JSONObject obj) throws JSONException;
    void load(JSONObject obj) throws JSONException;
}
