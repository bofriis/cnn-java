/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import com.improaim.org.json.JSONArray;
import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;

/**
 *
 * @author gola
 */
public class FullyConnLayer implements ILayer {
    
    int out_depth;

    // optional 
    double l1_decay_mul;
    double l2_decay_mul;

    // computed
    int num_inputs;
    int out_sx;
    int out_sy;
    String layer_type = "fc";

    public Vol[] filters;
    public Vol biases;
    
    Vol in_act;
    Vol out_act;
    
    Util global = new Util();
    
    public int get_out_sx() { return out_sx; }
    public int get_out_sy() { return out_sy; }
    public int get_out_depth() { return out_depth; }
    public String get_layer_type() { return layer_type; }
    public Vol get_out_act() { return out_act; }
    
    @Override
    public void set_in_act(Vol v) {
        this.in_act = v;
    }

    @Override
    public Vol get_in_act() {
        return this.in_act;
    }
    
    public FullyConnLayer(Definition opt) {
    
        // required
        // ok fine we will allow 'filters' as the word as well
        this.out_depth = opt.num_neurons != 0 ? opt.num_neurons : opt.filters;

        // optional 
        this.l1_decay_mul = opt.l1_decay_mul;
        this.l2_decay_mul = opt.l2_decay_mul != 0.0 ? opt.l2_decay_mul : 1.0;

        // computed
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = "fc";

        // initializations
        double bias = opt.bias_pref;
        this.filters = new Vol[this.out_depth];
        for(int i=0;i<this.out_depth ;i++) { 
            this.filters[i] = new Vol(1, 1, this.num_inputs, null); 
        }
        this.biases = new Vol(1, 1, this.out_depth, bias);
    
    }
    
    public Vol forward(Vol V, boolean is_training) {
      this.in_act = V;
      Vol A = new Vol(1, 1, this.out_depth, 0.0);
      double[] Vw = V.w;
      for(int i=0;i<this.out_depth;i++) {
        double a = 0.0;
        double[] wi = this.filters[i].w;
        for(int d=0;d<this.num_inputs;d++) {
          a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
        }
        a += this.biases.w[i];
        A.w[i] = a;
      }
      this.out_act = A;
      return this.out_act;
    }
    
    public void backward() {
      Vol V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out the gradient in input Vol
      
      // compute gradient wrt weights and data
      for(int i=0;i<this.out_depth;i++) {
        Vol tfi = this.filters[i];
        double chain_grad = this.out_act.dw[i];
        for(int d=0;d<this.num_inputs;d++) {
          V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
          tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params
        }
        this.biases.dw[i] += chain_grad;
      }
    }
    
    public ParamsAndGrads[] getParamsAndGrads() {
      ParamsAndGrads[] response = new ParamsAndGrads[this.out_depth + 1];
      for(int i=0;i<this.out_depth;i++) {
          response[i] = new ParamsAndGrads(this.filters[i].w, this.filters[i].dw, this.l1_decay_mul, this.l2_decay_mul);
      }
      response[this.out_depth] = new ParamsAndGrads(this.biases.w, this.biases.dw, 0.0, 0.0);
      
      return response;
    }
    

    @Override
    public void save(JSONObject obj) throws JSONException {
        obj.put("layer_type", this.layer_type);
        obj.put("out_depth", this.out_depth);
        obj.put("num_inputs", this.num_inputs);
        obj.put("out_sx", this.out_sx);
        obj.put("out_sy", this.out_sy);
        obj.put("l1_decay_mul", this.l1_decay_mul);
        obj.put("l2_decay_mul", this.l2_decay_mul);

        // Save filters as array of arrays
        JSONArray filtersArray = new JSONArray();
        for (Vol f : this.filters) {
            JSONArray weights = new JSONArray();
            for (double w : f.w) weights.put(w);
            filtersArray.put(weights);
        }
        obj.put("filters", filtersArray);

        // Save biases as array
        JSONArray biasesArray = new JSONArray();
        for (double b : this.biases.w) biasesArray.put(b);
        obj.put("biases", biasesArray);
    }

    @Override
    public void load(JSONObject obj) throws JSONException {
        this.layer_type = obj.getString("layer_type");
        this.out_depth = obj.getInt("out_depth");
        this.num_inputs = obj.getInt("num_inputs");
        this.out_sx = obj.getInt("out_sx");
        this.out_sy = obj.getInt("out_sy");
        this.l1_decay_mul = obj.getDouble("l1_decay_mul");
        this.l2_decay_mul = obj.getDouble("l2_decay_mul");

        // Load filters
        JSONArray filtersArray = obj.getJSONArray("filters");
        this.filters = new Vol[filtersArray.length()];
        for (int i = 0; i < filtersArray.length(); i++) {
            JSONArray weights = filtersArray.getJSONArray(i);
            Vol v = new Vol(1, 1, this.num_inputs, 0.0);
            for (int j = 0; j < weights.length(); j++) {
                v.w[j] = weights.getDouble(j);
            }
            this.filters[i] = v;
        }

        // Load biases
        JSONArray biasesArray = obj.getJSONArray("biases");
        this.biases = new Vol(1, 1, biasesArray.length(), 0.0);
        for (int i = 0; i < biasesArray.length(); i++) {
            this.biases.w[i] = biasesArray.getDouble(i);
        }
    }
}
