package rocks.propeller.karpathy;

import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;

public class RegressionLayer implements ILayer {
    int out_sx;
    int out_sy;
    int out_depth;
    int in_sx;
    int in_sy;
    int in_depth;
    String layer_type = "regression";

    Vol in_act;   // input
    Vol out_act;  // output
    Vol[] es;     // error

    public RegressionLayer(Definition opt) {
        this.out_sx = 1;
        this.out_sy = 1;
        this.out_depth = opt.num_neurons;
        this.in_sx = opt.in_sx;
        this.in_sy = opt.in_sy;
        this.in_depth = opt.in_depth;
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        this.out_act = V.cloneVol(); // identity
        return this.out_act;
    }

    // `y` is assumed to be the index of the target neuron, or should be a double[] for full targets
    public double backward(double[] y) {
        Vol x = this.in_act;
        Vol yhat = this.out_act;
        double loss = 0.0;

        for (int i = 0; i < this.out_depth; i++) {
            double dy = yhat.w[i] - y[i];
            x.dw[i] = dy;
            loss += 0.5 * dy * dy;
        }

        return loss;
    }

    // overload if needed to support single-value regression
    public double backward(double y) {
        Vol x = this.in_act;
        Vol yhat = this.out_act;

        double dy = yhat.w[0] - y;
        x.dw[0] = dy;

        return 0.5 * dy * dy;
    }

    @Override
    public void backward() {
        // nothing to do here; backward handled externally with real target
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0]; // no weights
    }

    @Override
    public int get_out_sx() {
        return this.out_sx;
    }

    @Override
    public int get_out_sy() {
        return this.out_sy;
    }

    @Override
    public int get_out_depth() {
        return this.out_depth;
    }

    @Override
    public String get_layer_type() {
        return this.layer_type;
    }

    @Override
    public Vol get_out_act() {
        return this.out_act;
    }

    @Override
    public void set_in_act(Vol v) {
        this.in_act = v;
    }

    @Override
    public Vol get_in_act() {
        return this.in_act;
    }
    
    @Override
    public void save(JSONObject json) throws JSONException {
        json.put("layer_type", layer_type);
        json.put("in_sx", in_sx);
        json.put("in_sy", in_sy);
        json.put("in_depth", in_depth);
        json.put("out_sx", out_sx);
        json.put("out_sy", out_sy);
        json.put("out_depth", out_depth);
    }

    @Override
    public void load(JSONObject json) throws JSONException {
        this.layer_type = json.getString("layer_type");
        this.in_sx = json.getInt("in_sx");
        this.in_sy = json.getInt("in_sy");
        this.in_depth = json.getInt("in_depth");
        this.out_sx = json.getInt("out_sx");
        this.out_sy = json.getInt("out_sy");
        this.out_depth = json.getInt("out_depth");
    }
}